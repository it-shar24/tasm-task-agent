"""Support-ticket triage agent.

Flow (one ticket):

    ticket -> LLM -> {"action":"call_tool"} -> Python validates + runs the tool -> result back to LLM -> ...
                  -> {"action":"finish"}    -> Python checks the decision      -> output dict

* The LLM chooses every tool call and when to stop (its instructions are in tam_agent/prompts.py).
* Python runs the tools (tam_agent/tools.py), refuses bad calls, enforces a call budget, checks the final
  decision against the evidence, and builds `reasoning_trace` from the calls that really happened.
* If the LLM cannot produce a valid decision, a clearly-marked low-confidence fallback is returned.
  Nothing is invented: unresolved accounts stay null, KB ids must come from real tool results.
"""
import json
import re
from dataclasses import dataclass, field

from . import prompts
from .llm import LLMClient
from .schema import CATEGORIES, PRIORITIES, ROUTES, validate
from .tools import TOOL_SPECS, Tools, summarize

MAX_TOOL_CALLS = 8        # tool budget per ticket
MAX_LLM_CALLS = 14        # hard stop for the whole conversation (tool calls + corrections + final answer)
MAX_STRIKES = 3           # consecutive unusable replies / failed tool calls before we give up
MAX_RESULT_CHARS = 6000   # tool results longer than this are cut before being shown to the LLM
MAX_TOKENS = 1800         # per LLM reply (the final decision contains a customer reply, so it is the longest)
ACCOUNT_TOOLS = {"get_account_context", "find_related_tickets"}   # tools that read one account's data


class Context:
    """Shared resources, built once per run: data, search indexes (inside Tools) and the LLM client."""

    def __init__(self, accounts, tickets, usage, kb):
        self.accounts = accounts
        self.tickets = tickets
        self.usage = usage
        self.kb = kb
        self.tools = Tools(accounts, tickets, usage, kb)
        self.llm = LLMClient()


# --------------------------------------------------------------------------- evidence
@dataclass
class Evidence:
    """What the tools have actually returned so far. Used to check the LLM's final answer."""
    account: dict | None = None                      # result of resolve_account
    context: dict | None = None                      # result of get_account_context
    policies_read: bool = False
    kb_seen: set = field(default_factory=set)        # KB ids whose content the LLM has seen
    related: dict = field(default_factory=dict)      # ticket_id -> candidate from find_related_tickets
    tools_ok: set = field(default_factory=set)       # names of tools that ran successfully

    @property
    def account_id(self):
        return self.account["account_id"] if self.account and self.account.get("status") == "resolved" else None

    def record(self, tool: str, result: dict):
        if "error" in result:
            return
        self.tools_ok.add(tool)
        if tool == "resolve_account":
            self.account = result
        elif tool == "get_account_context":
            self.context = result
        elif tool == "find_related_tickets":
            self.related.update({c["ticket_id"]: c for c in result["candidates"]})
        elif tool == "search_kb":
            self.kb_seen |= {r["id"] for r in result["results"]}
        elif tool == "read_kb_article":
            self.kb_seen.add(result["id"])
        elif tool == "get_policies":
            self.policies_read = True
            self.kb_seen |= {p["id"] for p in result["policies"]}


# --------------------------------------------------------------------------- parsing the LLM's reply
def parse_action(text: str):
    """Turn the LLM's reply into an action dict, or explain what is wrong. Returns (action, problem)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)          # tolerate code fences
    start = text.find("{")
    if start == -1:
        return None, "no JSON object found"
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])      # only the FIRST object is used
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc.msg}); keep customer_reply concise so the JSON is not cut off"
    if not isinstance(obj, dict):
        return None, "the JSON must be an object"

    kind = obj.get("action") or ("call_tool" if "tool" in obj else "finish" if "decision" in obj else None)
    if kind in ("call_tool", "tool_call", "tool"):
        if not isinstance(obj.get("tool"), str):
            return None, "a tool call needs a string field 'tool'"
        return {"action": "call_tool", "tool": obj["tool"], "args": obj.get("args") or {},
                "reason": str(obj.get("reason") or "")[:200]}, None
    if kind in ("finish", "final", "done"):
        if not isinstance(obj.get("decision"), dict):
            return None, "'finish' needs a 'decision' object"
        return {"action": "finish", "decision": obj["decision"]}, None
    return None, "'action' must be 'call_tool' or 'finish'"


# --------------------------------------------------------------------------- checking the decision
def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError


def normalize_decision(raw: dict):
    """Check types and allowed values of the LLM's decision. Returns (decision, errors)."""
    errors, d = [], {}

    def pick(key, allowed):
        if raw.get(key) in allowed:
            d[key] = raw[key]
        else:
            errors.append(f"{key} must be one of {allowed}")

    pick("category", CATEGORIES)
    pick("priority", PRIORITIES)
    pick("base_priority", PRIORITIES)
    pick("route_to", ROUTES)
    for key in ("escalate", "notify_tam", "needs_clarification"):
        try:
            d[key] = _as_bool(raw.get(key))
        except ValueError:
            errors.append(f"{key} must be true or false")
    dup = raw.get("duplicate_of")
    if dup in (None, "", "null", "None"):
        d["duplicate_of"] = None
    elif isinstance(dup, str) and re.fullmatch(r"TKT-\d+", dup.strip()):
        d["duplicate_of"] = dup.strip()
    else:
        errors.append("duplicate_of must be a ticket id like TKT-123456 or null")
    for key in ("kb_references", "account_signals"):
        value = raw.get(key, [])
        if isinstance(value, list):
            d[key] = [str(v).strip() for v in value if str(v).strip()]
        else:
            errors.append(f"{key} must be a list")
    for key in ("customer_reply", "internal_note"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            d[key] = raw[key].strip()
        else:
            errors.append(f"{key} must be a non-empty string")
    try:
        d["confidence"] = min(1.0, max(0.0, float(raw.get("confidence"))))    # clamp into 0..1
    except (TypeError, ValueError):
        errors.append("confidence must be a number between 0 and 1")
    d["rationale"] = str(raw.get("rationale") or "")[:300]
    return d, errors


def raised_priority(base: str) -> str:
    """KB-2914 account-risk override: raise one level, but never above P2 (P1/P2 stay as they are)."""
    rank = int(base[1])
    return base if rank <= 2 else f"P{rank - 1}"


def leaked_identifiers(reply: str, own_account_id, tools: Tools) -> list[str]:
    """Other customers' ids, domains or company names inside the customer reply (policy: never allowed)."""
    own = tools.accounts.get(own_account_id) if own_account_id else None
    text = reply.lower()
    if own:   # remove our own name/domain first so "Ironleaf Foods Europe" does not trip on "Ironleaf Foods"
        text = text.replace(own["company_name"].lower(), " ").replace(own["domain"].lower(), " ")
    found = [i for i in re.findall(r"ACC-\d{4}", reply.upper()) if i != own_account_id]
    found += [d for d in re.findall(r"[a-z0-9-]+\.example", text) if d in tools.by_domain]
    found += [a["company_name"] for a in tools.accounts.values()
              if a["account_id"] != own_account_id and a["company_name"].lower() in text]
    return sorted(set(found))


def apply_guardrails(d: dict, ev: Evidence) -> list[str]:
    """Deterministic fixes for rules the support policy states unambiguously. Edits `d`; returns what changed.

    The LLM makes the judgement calls; these only enforce policy text from KB-2914 / KB-5930 / KB-6401, and
    every change is written to reasoning_trace.
    """
    notes = []

    def set_(key, value, why):
        if d[key] != value:
            notes.append(f"{key}: {d[key]} -> {value} ({why})")
            d[key] = value

    refs = list(dict.fromkeys(d["kb_references"]))                  # remove duplicates, keep order
    kept = [r for r in refs if r in ev.kb_seen]
    if kept != refs:
        notes.append(f"dropped kb_references never retrieved by a tool: {[r for r in refs if r not in kept]}")
    d["kb_references"] = kept

    if d["category"] == "security":                                  # KB-2914 / KB-6401
        set_("priority", "P1", "security reports are P1")
        set_("route_to", "security", "security reports go to Security, never L1")
        set_("escalate", True, "security reports are escalated")
    elif d["duplicate_of"]:                                          # KB-5930
        set_("category", "duplicate", "duplicate_of is set")
        set_("escalate", False, "duplicates are not re-escalated")

    risk = (ev.context or {}).get("risk_override", {})
    if risk.get("applies") and d["category"] != "security":          # KB-2914 account risk override
        if not d["duplicate_of"]:
            set_("priority", raised_priority(d["base_priority"]), "risk override: raise one level, max P2")
            set_("escalate", True, "risk override is escalated")
        set_("notify_tam", True, "risk override notifies the TAM")

    if d["category"] == "needs_info":                                # KB-2914 insufficient information
        set_("needs_clarification", True, "category is needs_info")
        if not risk.get("applies"):
            set_("priority", "P3", "insufficient information is P3")
    if ev.account_id is None:
        set_("needs_clarification", True, "account could not be resolved")

    if d["priority"] == "P1" and not d["duplicate_of"]:              # KB-2914 escalate flag
        set_("escalate", True, "P1 is escalated")
    return notes


# --------------------------------------------------------------------------- the agent run
class TriageRun:
    """State for one ticket: conversation, evidence, trace."""

    def __init__(self, ticket: dict, ctx: Context):
        self.ticket, self.ctx = ticket, ctx
        self.ev = Evidence()
        self.trace: list[dict] = []
        self.tool_calls = 0
        self.nudged = False          # the one-time "you skipped something important" reminder

    # ---- trace
    def _add(self, action: str, input_: str, observation: str, reason: str = ""):
        entry = {"step": len(self.trace) + 1, "action": action, "input": input_, "observation": observation}
        if reason:
            entry["reason"] = reason
        self.trace.append(entry)

    # ---- main loop
    def execute(self) -> dict:
        messages = [{"role": "user", "content": prompts.ticket_message(self.ticket)}]
        system, strikes = prompts.system_prompt(), 0
        for _ in range(MAX_LLM_CALLS):
            reply = self._ask(messages, system)
            messages.append({"role": "assistant", "content": reply})
            action, problem = parse_action(reply)
            if problem:
                self._add("malformed_llm_response", reply[:100].replace("\n", " "), problem)
                feedback, strikes = prompts.malformed_message(problem), strikes + 1
            elif action["action"] == "call_tool":
                feedback, ok = self._tool_step(action)
                strikes = 0 if ok else strikes + 1
            else:
                output, feedback = self._finish_step(action["decision"])
                if output is not None:
                    return output
                strikes += 1
            if strikes >= MAX_STRIKES:
                return self._fallback(f"{MAX_STRIKES} unusable steps in a row")
            messages.append({"role": "user", "content": feedback})
        return self._fallback(f"no valid decision within {MAX_LLM_CALLS} LLM calls")

    def _ask(self, messages, system) -> str:
        last_error = None
        for _ in range(2):                                           # one retry for transient API errors
            try:
                return self.ctx.llm.chat(messages, system=system, max_tokens=MAX_TOKENS, temperature=0.0,
                                         json_mode=True)
            except Exception as exc:
                last_error = exc
        # Not a triage problem we can solve: raise so run.py logs an exception instead of a fake result.
        raise RuntimeError(f"LLM call failed for {self.ticket.get('ticket_id')}: {last_error!r}") from last_error

    # ---- a tool call requested by the LLM
    def _tool_step(self, action):
        """Validate and run one tool call. Returns (message for the LLM, succeeded?)."""
        name, args, reason = action["tool"], action["args"], action["reason"]
        shown_args = json.dumps(args, ensure_ascii=False)[:300] if isinstance(args, dict) else str(args)[:300]
        if self.tool_calls >= MAX_TOOL_CALLS:
            self._add(f"rejected_tool_call:{name}", shown_args, f"tool budget of {MAX_TOOL_CALLS} used up", reason)
            return prompts.budget_exhausted_message(MAX_TOOL_CALLS), False
        self.tool_calls += 1

        if name in ACCOUNT_TOOLS and isinstance(args, dict):         # never read another customer's data
            account_id = self.ev.account_id
            args = {**args, "account_id": args.get("account_id") or account_id}
            if account_id is None:
                result = {"error": "the account is not resolved; call resolve_account first. If it stays "
                                   "unresolved, account tools cannot be used"}
            elif str(args["account_id"]).strip().upper() != account_id:
                result = {"error": f"this ticket belongs to {account_id}; account tools may only use that id"}
            else:
                result = self.ctx.tools.call(name, args, self.ticket)
        else:
            result = self.ctx.tools.call(name, args, self.ticket)

        self.ev.record(name, result)
        known = name in TOOL_SPECS
        self._add(name if known else f"invalid_tool:{name}", shown_args,
                  summarize(name, args, result) if known else result["error"], reason)
        return (prompts.tool_result_message(name, result, self.tool_calls, MAX_TOOL_CALLS, MAX_RESULT_CHARS),
                "error" not in result)

    # ---- the LLM says it is done
    def _finish_step(self, raw_decision):
        """Check the decision. Returns (output, None) if accepted, else (None, feedback for the LLM)."""
        d, errors = normalize_decision(raw_decision)
        if errors:
            self._add("rejected_decision", "finish", "; ".join(errors))
            return None, "Your decision was rejected: " + "; ".join(errors) + ". Send a corrected finish."

        if "resolve_account" not in self.ev.tools_ok:
            self._add("rejected_decision", "finish", "account was never resolved")
            return None, ("Call resolve_account before finishing: the output needs the resolved account "
                          "(or an honest 'unresolved').")

        skipped = []
        if not self.ev.policies_read:
            skipped.append("get_policies (priority, routing and escalation must come from the policy)")
        if self.ev.account_id and self.ev.context is None:
            skipped.append("get_account_context (the policy's account-risk override depends on it)")
        if skipped and not self.nudged:                               # remind once; the LLM may then still finish
            self.nudged = True
            self._add("reminder", "finish", "not yet called: " + "; ".join(skipped))
            return None, ("Before finishing, consider: you have not called " + "; ".join(skipped) +
                          ". Call them, or send your finish again if you are sure they are unnecessary.")

        problems = self._evidence_problems(d)
        if problems:
            self._add("rejected_decision", "finish", "; ".join(problems))
            return None, "Your decision was rejected: " + "; ".join(problems) + ". Send a corrected finish."

        for note in apply_guardrails(d, self.ev):
            self._add("guardrail", "policy check", note)
        return self._build_output(d), None

    def _evidence_problems(self, d) -> list[str]:
        """Claims in the decision that the tool results do not support."""
        problems = []
        dup = d["duplicate_of"]
        if d["category"] == "duplicate" and not dup:
            problems.append("category 'duplicate' needs duplicate_of; otherwise choose another category")
        if dup:
            cand = self.ev.related.get(dup)
            if cand is None:
                problems.append(f"duplicate_of {dup} was not returned by find_related_tickets for this account")
            elif not cand["meets_hard_duplicate_criteria"]:
                failed = [k for k, v in cand["checks"].items() if v is False or (k == "same_module" and v is None)]
                problems.append(f"{dup} fails the duplicate policy checks {failed}; set duplicate_of to null")
        leaks = leaked_identifiers(d["customer_reply"], self.ev.account_id, self.ctx.tools)
        if leaks:
            problems.append(f"customer_reply mentions other customers/accounts {leaks}; remove them")
        return problems

    # ---- final output
    def _build_output(self, d: dict) -> dict:
        self._add("finalize", "decision",
                  f"category={d['category']}, priority={d['priority']} (base {d['base_priority']}), "
                  f"route={d['route_to']}, escalate={d['escalate']}, notify_tam={d['notify_tam']}, "
                  f"duplicate_of={d['duplicate_of']}, needs_clarification={d['needs_clarification']}, "
                  f"kb={d['kb_references']}, confidence={d['confidence']:.2f}", d["rationale"])
        out = {
            "ticket_id": self.ticket["ticket_id"],
            "account_id": self.ev.account_id,                         # from the tool, never from the LLM
            "category": d["category"], "priority": d["priority"], "route_to": d["route_to"],
            "escalate": d["escalate"], "notify_tam": d["notify_tam"], "duplicate_of": d["duplicate_of"],
            "needs_clarification": d["needs_clarification"], "kb_references": d["kb_references"],
            "account_signals": d["account_signals"], "customer_reply": d["customer_reply"],
            "internal_note": d["internal_note"], "confidence": d["confidence"], "reasoning_trace": self.trace,
        }
        errors = validate(out)
        if errors:
            return self._fallback("final output failed schema validation: " + "; ".join(errors))
        return out

    def _fallback(self, reason: str) -> dict:
        """Safe, honest result when the LLM could not finish. It makes no classification claims."""
        if "resolve_account" not in self.ev.tools_ok:                 # a real lookup, so nothing is fabricated
            t = self.ticket
            args = {"account_id": t.get("account_id", ""), "requester_domain": t.get("requester_domain", ""),
                    "company_name": t.get("company_name_raw", "")}
            result = self.ctx.tools.call("resolve_account", args, t)
            self.ev.record("resolve_account", result)
            self._add("resolve_account", json.dumps(args), summarize("resolve_account", args, result), "fallback")
        self._add("fallback", "agent stopped", reason)
        return {
            "ticket_id": self.ticket["ticket_id"], "account_id": self.ev.account_id,
            "category": "needs_info", "priority": "P3", "route_to": "L1_support", "escalate": False,
            "notify_tam": False, "duplicate_of": None, "needs_clarification": True, "kb_references": [],
            "account_signals": [],
            "customer_reply": ("Thank you for contacting us. We have received your request and a support engineer "
                               "will review it shortly. To help us, please share the module affected, the exact error "
                               "text, how many users are affected and when it started.\n\nSupport Team"),
            "internal_note": f"AUTOMATED TRIAGE INCOMPLETE ({reason}). No classification was made; "
                             "manual triage required.",
            "confidence": 0.1, "reasoning_trace": self.trace,
        }


def triage(ticket: dict, ctx: Context) -> dict:
    """Triage one ticket. `ticket` is a row of tickets.csv as a dict; returns a dict matching schema.py."""
    return TriageRun(ticket, ctx).execute()
