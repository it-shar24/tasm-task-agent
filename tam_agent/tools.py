"""The lookup tools the agent can call.

Design rule: the LLM does the *judgement*, Python does the *lookups and arithmetic*.
Each tool therefore returns small, already-computed facts (days to renewal, % change in
users, whether a ticket meets the duplicate-policy tests) instead of raw tables.

Tools never raise into the agent loop: bad input comes back as {"error": "..."} so the
LLM can correct itself.
"""
from collections import defaultdict
from datetime import date, datetime

from .data import AS_OF_DATE
from .retrieval import (BM25, CANONICAL_MODULES, canonical_module, clean_domain, extract_error_code, jaccard,
                        name_similarity, normalize_company, tokenize)

AS_OF = date.fromisoformat(AS_OF_DATE)

# Thresholds below are NOT invented: they are copied from the support policy articles.
# tests/test_tools.py checks that the policy text still contains these numbers.
RISK_RENEWAL_DAYS = 90        # KB-2914 "Account risk override": within 90 days of renewal
RISK_USER_DECLINE_PCT = 25    # KB-2914: active users declined more than 25% over the last 3 months
RISK_ESCALATIONS = 2          # KB-2914: 2 or more open escalations
DUPLICATE_WINDOW_DAYS = 7     # KB-5930: same account/module/code or symptom within the previous 7 days
OPEN_STATUSES = {"open", "pending_customer"}   # KB-5930: "open or pending"

EXACT_CODE_BOOST = 10.0       # an article listing the ticket's error code should outrank text matches
MODULE_MATCH_BOOST = 1.3
EXCERPT_CHARS = 450
ARTICLE_CHARS = 2500

# What the LLM is shown. `args` maps argument name -> description; `required` must be non-empty.
TOOL_SPECS = {
    "resolve_account": {
        "description": "Identify which customer account this ticket belongs to. Uses account_id, requester_domain "
                       "(most reliable) and company name. Returns status resolved / uncertain / unresolved. "
                       "Never guesses from similar-looking names.",
        "args": {"account_id": "account id from the ticket (may be blank)",
                 "requester_domain": "requester_domain from the ticket",
                 "company_name": "company_name_raw from the ticket"},
        "required": [],
    },
    "get_account_context": {
        "description": "Facts about the resolved account: tier, renewal date and days left, open escalations, "
                       "licensed modules, SSO change date, usage trend (last months, % change in active users), "
                       "the policy risk-override check, and recent ticket history with statuses.",
        "args": {"account_id": "the account_id returned by resolve_account"},
        "required": ["account_id"],
    },
    "find_related_tickets": {
        "description": "Recent tickets (last 30 days, before this one) from the SAME account, ranked by similarity, "
                       "each with the duplicate-policy checks already computed. Use to detect duplicates or "
                       "repeat issues.",
        "args": {"account_id": "the account_id returned by resolve_account",
                 "module": "optional module name to compare (defaults to this ticket's module)",
                 "error_code": "optional error code to compare (defaults to this ticket's code)",
                 "query": "optional symptom words to compare (defaults to this ticket's subject + description)"},
        "required": ["account_id"],
    },
    "search_kb": {
        "description": "Search the knowledge base (troubleshooting, how-to, limits, known issues, reference, FAQ). "
                       "Deprecated articles are replaced by their current version. Policy articles are not "
                       "searched here; use get_policies.",
        "args": {"query": "keywords: symptom, feature, error text",
                 "module": "optional module to prefer (e.g. eInvoice)",
                 "error_code": "optional error code, e.g. E-4102 (exact matches rank first)",
                 "top_k": "optional number of results, default 5, max 8"},
        "required": ["query"],
    },
    "read_kb_article": {
        "description": "Read one KB article in full (use to follow a `related` link or to confirm details). "
                       "Deprecated articles are followed to their replacement.",
        "args": {"article_id": "e.g. KB-4795"},
        "required": ["article_id"],
    },
    "get_policies": {
        "description": "Return the support policy articles: priority definitions, account-risk override, routing, "
                       "escalation, duplicate rules, data-exposure handling, insufficient-information rule.",
        "args": {},
        "required": [],
    },
}


def _created(row) -> datetime:
    return datetime.fromisoformat(row["created_at"])


class Tools:
    """Builds all indexes once; every lookup afterwards is a dictionary/list access."""

    def __init__(self, accounts, tickets, usage, kb):
        # --- accounts: by id, by domain, by normalised legal name
        self.accounts = {r["account_id"]: r for r in accounts.to_dict("records")}
        self.by_domain = {r["domain"].lower(): r["account_id"] for r in self.accounts.values()}
        self.by_name = defaultdict(list)
        for r in self.accounts.values():
            self.by_name[normalize_company(r["company_name"])].append(r["account_id"])

        # --- tickets: add clean helper fields. A blank account_id is filled from the domain so a
        # blank-id ticket still counts as history for its real account.
        self.tickets = {}
        self.tickets_by_account = defaultdict(list)
        for r in tickets.to_dict("records"):
            r["eff_account"] = r["account_id"] or self.by_domain.get(r["requester_domain"].lower(), "")
            r["module_canon"] = canonical_module(r["module"])
            r["code"] = r["error_code"] or extract_error_code(r["subject"], r["description"])
            r["created"] = _created(r)
            r["tokens"] = tokenize(r["subject"] + " " + r["description"])
            self.tickets[r["ticket_id"]] = r
            if r["eff_account"]:
                self.tickets_by_account[r["eff_account"]].append(r)
        for rows in self.tickets_by_account.values():
            rows.sort(key=lambda r: r["created"])

        # --- usage: per account, oldest -> newest
        self.usage = {acc: g.sort_values("month").to_dict("records") for acc, g in usage.groupby("account_id")}

        # --- knowledge base: policies are kept separate (get_policies); the rest is BM25-indexed
        self.kb = {a["id"]: a for a in kb}
        self.policies = [a for a in kb if a["meta"].get("type") == "policy" and a["meta"].get("status") == "current"]
        self.search_articles = [a for a in kb if a["meta"].get("type") != "policy"]
        self.bm25 = BM25([self._article_tokens(a) for a in self.search_articles])

    @staticmethod
    def _article_tokens(article) -> list[str]:
        m = article["meta"]
        # the title is repeated so title words count more than body words
        text = " ".join([m.get("title", "")] * 3 + [m.get("module", ""), " ".join(m.get("error_codes", [])), article["body"]])
        return tokenize(text)

    # ------------------------------------------------------------------ dispatcher
    def call(self, name: str, args, ticket: dict) -> dict:
        """Validate the LLM's tool request in Python, then run it. Always returns a dict."""
        spec = TOOL_SPECS.get(name)
        if spec is None:
            return {"error": f"unknown tool '{name}'. Available tools: {', '.join(TOOL_SPECS)}"}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"error": "args must be a JSON object"}
        unexpected = sorted(set(args) - set(spec["args"]))
        if unexpected:
            return {"error": f"unexpected argument(s) {unexpected} for {name}; allowed: {list(spec['args'])}"}
        missing = [a for a in spec["required"] if not str(args.get(a) or "").strip()]
        if missing:
            return {"error": f"missing required argument(s) {missing} for {name}"}
        try:
            return getattr(self, name)(ticket, **args)
        except Exception as exc:  # a bug or odd input must not crash the whole run
            return {"error": f"{name} failed: {exc!r}"}

    # ------------------------------------------------------------------ A. account resolution
    def resolve_account(self, ticket, account_id="", requester_domain="", company_name=""):
        acc_id = str(account_id or "").strip().upper()
        domain = clean_domain(str(requester_domain or ""))
        name = str(company_name or "").strip()
        if not (acc_id or domain or name):
            return {"status": "unresolved", "account_id": None, "note": "no account_id, domain or company name given"}

        by_id = acc_id if acc_id in self.accounts else None
        by_domain = self.by_domain.get(domain) if domain else None
        notes = []
        if acc_id and not by_id:
            notes.append(f"account_id '{acc_id}' does not exist")

        if by_id and by_domain and by_id != by_domain:   # identifiers disagree: do not pick one
            return {"status": "uncertain", "account_id": None,
                    "note": f"account_id {by_id} and domain {domain} point to different accounts ({by_domain})",
                    "candidates": [self._brief(by_id), self._brief(by_domain)]}
        if by_id or by_domain:
            chosen = by_id or by_domain
            method = "account_id+domain" if (by_id and by_domain) else ("account_id" if by_id else "requester_domain")
            return self._resolved(chosen, method, "high", name, notes)

        # no usable id/domain: only an EXACT normalised name match on a single account is accepted
        exact = self.by_name.get(normalize_company(name), []) if name else []
        if len(exact) == 1:
            return self._resolved(exact[0], "company_name_exact", "medium", name, notes)
        if len(exact) > 1:
            return {"status": "uncertain", "account_id": None, "note": "several accounts share this exact name",
                    "candidates": [self._brief(a) for a in exact]}
        similar = self._similar_names(name, exclude=None)
        if similar:   # looks like something, but similar is not good enough to choose
            return {"status": "uncertain", "account_id": None,
                    "note": "no id/domain match and no exact name match; similar names are NOT evidence",
                    "candidates": similar}
        return {"status": "unresolved", "account_id": None, "note": "; ".join(notes) or "no matching account"}

    def _brief(self, account_id):
        a = self.accounts[account_id]
        return {"account_id": account_id, "company_name": a["company_name"], "domain": a["domain"]}

    def _similar_names(self, name, exclude, threshold=0.75, limit=3):
        scored = [(name_similarity(name, a["company_name"]), a["account_id"]) for a in self.accounts.values()
                  if a["account_id"] != exclude]
        return [self._brief(i) | {"name_similarity": round(s, 2)}
                for s, i in sorted(scored, reverse=True)[:limit] if s >= threshold]

    def _resolved(self, account_id, method, confidence, raw_name, notes):
        a = self.accounts[account_id]
        out = {"status": "resolved", "account_id": account_id, "company_name": a["company_name"],
               "tier": a["tier"], "method": method, "confidence": confidence}
        if raw_name and normalize_company(raw_name) != normalize_company(a["company_name"]):
            notes.append(f"typed company name '{raw_name}' differs from legal name; {method} match is authoritative")
        # only names close enough to be confused with the chosen one (0.88) are worth flagging
        lookalikes = self._similar_names(raw_name or a["company_name"], exclude=account_id, threshold=0.88)
        if lookalikes:
            out["lookalike_accounts_not_selected"] = [f"{x['account_id']} {x['company_name']}" for x in lookalikes]
        if notes:
            out["note"] = "; ".join(notes)
        return out

    # ------------------------------------------------------------------ B. account context
    def get_account_context(self, ticket, account_id):
        acc = self.accounts.get(str(account_id).strip().upper())
        if acc is None:
            return {"error": f"unknown account_id '{account_id}'. Use the id returned by resolve_account."}
        acc_id = acc["account_id"]
        days_to_renewal = (date.fromisoformat(acc["renewal_date"]) - AS_OF).days
        escalations = int(acc["open_escalations"] or 0)

        rows = self.usage.get(acc_id, [])
        months = [{"month": r["month"], "active_users": int(r["active_users"]), "login_rate_pct": float(r["login_rate_pct"]),
                   "tickets_opened": int(r["tickets_opened"])} for r in rows[-6:]]
        change_pct = None
        if len(rows) >= 4 and rows[-4]["active_users"]:      # latest month vs 3 months earlier
            first, last = rows[-4]["active_users"], rows[-1]["active_users"]
            change_pct = round((last - first) / first * 100, 1)
        usage = {"last_months": months, "active_users_change_3m_pct": change_pct,
                 "compared": f"{rows[-4]['month']} -> {rows[-1]['month']}" if len(rows) >= 4 else None}

        risk = {"within_90_days_of_renewal": 0 <= days_to_renewal <= RISK_RENEWAL_DAYS,
                "active_users_down_over_25pct": change_pct is not None and change_pct < -RISK_USER_DECLINE_PCT,
                "two_or_more_open_escalations": escalations >= RISK_ESCALATIONS}
        risk["applies"] = risk["within_90_days_of_renewal"] and (risk["active_users_down_over_25pct"]
                                                                  or risk["two_or_more_open_escalations"])
        risk["rule"] = ("KB-2914: raise priority one level (max P2), notify TAM, escalate"
                        if risk["applies"] else "KB-2914 account risk override does not apply")

        sso_days = None
        if acc["sso_enabled_date"]:
            sso_days = (AS_OF - date.fromisoformat(acc["sso_enabled_date"])).days

        history = self._history(acc_id, ticket)
        facts = [f"{acc['tier']} tier", f"renewal {acc['renewal_date']} ({days_to_renewal} days)",
                 f"{escalations} open escalation(s)"]
        if change_pct is not None:
            facts.append(f"active users {change_pct:+.1f}% over 3 months ({usage['compared']})")
        if sso_days is not None:
            facts.append(f"SSO enabled/reconfigured {acc['sso_enabled_date']} ({sso_days} days ago)")
        return {"account_id": acc_id, "company_name": acc["company_name"], "tier": acc["tier"], "region": acc["region"],
                "modules_licensed": acc["modules_licensed"].split(";"), "renewal_date": acc["renewal_date"],
                "days_to_renewal": days_to_renewal, "open_escalations": escalations,
                "sso_enabled_date": acc["sso_enabled_date"] or None, "days_since_sso_change": sso_days,
                "nps_last": acc["nps_last"] or None, "usage": usage, "risk_override": risk,
                "ticket_history": history, "signal_facts": facts}

    def _history(self, account_id, ticket):
        ref = _created(ticket)
        past = [t for t in self.tickets_by_account.get(account_id, [])
                if t["ticket_id"] != ticket["ticket_id"] and 0 <= (ref - t["created"]).days <= 90]
        by_status = defaultdict(int)
        for t in past:
            by_status[t["status"]] += 1
        recent = [f"{t['ticket_id']} {t['created_at'][:10]} {t['module_canon'] or t['module'] or '?'} "
                  f"{t['code'] or '-'} [{t['status']}] {t['subject'][:60]}" for t in past[-5:]]
        return {"last_90_days_by_status": dict(by_status), "most_recent": recent}

    # ------------------------------------------------------------------ D. related tickets / duplicates
    def find_related_tickets(self, ticket, account_id, module="", error_code="", query=""):
        acc_id = str(account_id).strip().upper()
        if acc_id not in self.accounts:
            return {"error": f"unknown account_id '{account_id}'. Use the id returned by resolve_account."}
        ref = _created(ticket)
        my_module = canonical_module(module) or (module if module in CANONICAL_MODULES else None) \
            or canonical_module(ticket["module"])
        my_code = (error_code or "").strip().upper() or extract_error_code(ticket["subject"], ticket["description"]) \
            or ticket.get("error_code") or None
        my_tokens = tokenize(query) if query else tokenize(ticket["subject"] + " " + ticket["description"])

        cands = []
        for t in self.tickets_by_account.get(acc_id, []):
            age_days = (ref - t["created"]).total_seconds() / 86400
            if t["ticket_id"] == ticket["ticket_id"] or not 0 <= age_days <= 30:
                continue
            same_module = (t["module_canon"] == my_module) if (t["module_canon"] and my_module) else None
            same_code = (t["code"] == my_code) if (t["code"] and my_code) else None
            checks = {"within_7_days": age_days <= DUPLICATE_WINDOW_DAYS,
                      "open_or_pending": t["status"] in OPEN_STATUSES,
                      "same_module": same_module, "same_error_code": same_code}
            hard = (checks["within_7_days"] and checks["open_or_pending"] and same_module is True
                    and same_code is not False)
            cands.append({"ticket_id": t["ticket_id"], "created_at": t["created_at"], "days_before": round(age_days, 1),
                          "module": t["module_canon"] or t["module"] or None, "error_code": t["code"],
                          "status": t["status"], "customer_priority": t["customer_priority"] or None,
                          "subject": t["subject"], "description_start": t["description"][:160],
                          "text_similarity": round(jaccard(my_tokens, t["tokens"]), 2),
                          "checks": checks, "meets_hard_duplicate_criteria": hard})
        cands.sort(key=lambda c: (c["meets_hard_duplicate_criteria"], c["checks"]["same_error_code"] is True,
                                  c["checks"]["same_module"] is True, c["text_similarity"]), reverse=True)
        return {"account_id": acc_id, "compared_module": my_module, "compared_error_code": my_code,
                "candidates": cands[:6],
                "how_to_read": "Duplicate (KB-5930) needs meets_hard_duplicate_criteria=true AND the same error code "
                               "or the same symptom (you judge symptom from subject/description). Similar wording "
                               "alone is not proof; a different error code means a different issue."}

    # ------------------------------------------------------------------ C. knowledge base
    def search_kb(self, ticket, query, module="", error_code="", top_k=5):
        try:
            top_k = max(1, min(int(top_k), 8))
        except (TypeError, ValueError):
            top_k = 5
        code = (error_code or "").strip().upper() or extract_error_code(query)
        mod = canonical_module(module) or (module if module in CANONICAL_MODULES else None)
        scores = self.bm25.scores(tokenize(query))
        ranked = []
        for art, s in zip(self.search_articles, scores):
            meta = art["meta"]
            if code and code in meta.get("error_codes", []):
                s += EXACT_CODE_BOOST
            if mod and meta.get("module") == mod:
                s *= MODULE_MATCH_BOOST
            if s > 0:
                ranked.append((s, art))
        ranked.sort(key=lambda x: -x[0])

        results, seen = [], set()
        for score, art in ranked:
            final, note = self._follow_supersession(art)
            if final is None:
                continue                                  # deprecated with no usable replacement
            if final["id"] in seen:
                continue
            seen.add(final["id"])
            results.append(self._brief_article(final, round(score, 2), note))
            if len(results) == top_k:
                break
        return {"query": query, "results": results} if results else {"query": query, "results": [],
                                                                      "note": "no matching articles; try other words"}

    def _follow_supersession(self, art):
        """Deprecated article -> its replacement (up to 3 hops). Returns (article, note) or (None, None)."""
        note, hops = None, 0
        while art["meta"].get("status") == "deprecated":
            target = self.kb.get(art["meta"].get("superseded_by", ""))
            hops += 1
            if target is None or hops > 3:
                return None, None
            note = f"{(note + '; ') if note else ''}{art['id']} is deprecated, replaced by {target['id']}"
            art = target
        if art["meta"].get("status") == "resolved":
            note = (note + "; " if note else "") + "this incident is marked resolved (historical)"
        return art, note

    def _brief_article(self, art, score, note):
        m = art["meta"]
        body = " ".join(art["body"].split("\n", 2)[-1].split())      # drop the '# title' line, collapse spaces
        out = {"id": art["id"], "title": m.get("title"), "type": m.get("type"), "status": m.get("status"),
               "module": m.get("module"), "error_codes": m.get("error_codes", []), "score": score,
               "excerpt": body[:EXCERPT_CHARS], "related": m.get("related", [])}
        if note:
            out["note"] = note
        return out

    def read_kb_article(self, ticket, article_id):
        art = self.kb.get(str(article_id).strip().upper())
        if art is None:
            return {"error": f"no article '{article_id}'. Use ids from search results or `related` lists."}
        final, note = self._follow_supersession(art)
        if final is None:
            return {"error": f"{article_id} is deprecated and has no usable replacement"}
        m = final["meta"]
        related = []
        for rid in m.get("related", []):
            r = self.kb.get(rid)
            if r:
                item = {"id": rid, "title": r["meta"].get("title"), "status": r["meta"].get("status")}
                if r["meta"].get("superseded_by"):
                    item["superseded_by"] = r["meta"]["superseded_by"]
                related.append(item)
        body = final["body"].strip()
        out = {"id": final["id"], "title": m.get("title"), "type": m.get("type"), "status": m.get("status"),
               "module": m.get("module"), "error_codes": m.get("error_codes", []),
               "last_updated": m.get("last_updated"), "related": related,
               "content": body[:ARTICLE_CHARS], "truncated": len(body) > ARTICLE_CHARS}
        if note:
            out["note"] = note
        return out

    # ------------------------------------------------------------------ E. policy lookup
    def get_policies(self, ticket):
        return {"policies": [{"id": p["id"], "title": p["meta"].get("title"), "content": p["body"].strip()}
                             for p in self.policies]}


# ---------------------------------------------------------------------- trace summaries
def summarize(name: str, args: dict, result: dict) -> str:
    """One short, factual line per tool result for reasoning_trace (built from the real result)."""
    if "error" in result:
        return f"error: {result['error']}"
    if name == "resolve_account":
        if result["status"] == "resolved":
            extra = f"; lookalikes not selected: {result['lookalike_accounts_not_selected']}" \
                if result.get("lookalike_accounts_not_selected") else ""
            return f"resolved {result['account_id']} {result['company_name']} via {result['method']}{extra}"
        return f"{result['status']}: {result.get('note', '')}"
    if name == "get_account_context":
        r = result["risk_override"]
        return (f"{'; '.join(result['signal_facts'])}; risk_override={'APPLIES' if r['applies'] else 'no'} "
                f"(renewal<=90d={r['within_90_days_of_renewal']}, users_down>25%={r['active_users_down_over_25pct']}, "
                f"escalations>=2={r['two_or_more_open_escalations']}); history 90d={result['ticket_history']['last_90_days_by_status']}")
    if name == "find_related_tickets":
        c = result["candidates"]
        if not c:
            return "no tickets from this account in the previous 30 days"
        top = "; ".join(f"{x['ticket_id']} [{x['status']}, {x['days_before']}d, module={x['module']}, code={x['error_code']}, "
                        f"hard_criteria={x['meets_hard_duplicate_criteria']}]" for x in c[:3])
        return f"{len(c)} candidate(s): {top}"
    if name == "search_kb":
        r = result["results"]
        return "; ".join(f"{x['id']} ({x['type']}/{x['status']}) {x['title'][:60]}" for x in r) or result.get("note", "no results")
    if name == "read_kb_article":
        return (f"read {result['id']} ({result['type']}/{result['status']}) {result['title'][:70]}"
                f"{'; ' + result['note'] if result.get('note') else ''}; related={[x['id'] for x in result['related']]}")
    if name == "get_policies":
        return "read policies: " + ", ".join(f"{p['id']} {p['title']}" for p in result["policies"])
    return "ok"
