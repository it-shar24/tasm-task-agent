# SOLUTION

## 1. Design
`ticket → LLM → tool call → Python validates + runs it → result → LLM → … → finish → Python checks → output`

A small custom loop (`tam_agent/agent.py`), no framework. `llm.chat()` is text-only, so the LLM replies with one JSON object: a tool call or a final decision. Python validates tool names and arguments before running anything.

**Tools** (`tools.py`; Python does lookups and arithmetic, the LLM judges): `resolve_account` (id, domain, then *exact* normalised name; similar names return `uncertain`, never a guess) · `get_account_context` (tier, days to renewal vs 2026-10-20, 3-month active-user change, escalations, SSO date, history, KB-2914 risk-override test) · `find_related_tickets` (same account, last 30 days, blank-id tickets linked by domain, KB-5930 checks precomputed) · `search_kb` (BM25 + error-code/module boosts; deprecated → `superseded_by`) · `read_kb_article` (follow `related`) · `get_policies`.

**Choosing and stopping.** The LLM picks each tool, its arguments and when to finish. Limits: 8 tool calls, 14 LLM calls, 3 unusable replies in a row → fallback. It cannot finish before `resolve_account`; if it skipped `get_policies` or `get_account_context` it is reminded once and may still finish. Account tools accept only the resolved account.

**Validation.** Types/enums are checked; KB ids must come from real tool results; `duplicate_of` must be a candidate meeting KB-5930; replies may not name other customers. Unambiguous policy text is enforced in code and logged in the trace (security → P1/Security/escalate; risk override → +1 level, max P2, TAM, escalate; duplicates not re-escalated; P1 escalated; insufficient info → P3). `account_id` and `reasoning_trace` come from real tool calls, never LLM text. Ticket text is framed as untrusted. Output passes `schema.validate`. If the LLM cannot finish, the fallback is a low-confidence `needs_info` marked "manual triage required"; if the LLM is down, the run logs an exception.

## 2. Evaluation
- ~300-ticket gold set labelled by senior engineers, oversampling hard cases (blank ids, lookalike names, incidents, security, duplicates, vague tickets).
- Account resolution accuracy, plus wrong-account rate separately (wrong is worse than null).
- Priority, route, escalate, notify_tam: accuracy and confusion matrix; P1 under-triage and security sent to L1 are critical failures; rule-based policy-compliance check.
- Duplicates: precision first (a false duplicate hides a real ticket), then recall.
- KB: recall@5, ungrounded or deprecated citations (target 0), correct `related` follow-ups.
- Replies: human rubric, automated checks for other-customer/internal data, prompt-injection suite.
- Ops: LLM calls, p50/p95 latency, cost per ticket, fallback rate, run-to-run consistency; shadow mode beside humans before auto-routing.

## 3. Scale (30,000 tickets, 4,000 articles)
Indexes are built once in memory (0.3 s today). Tool calls are cheap; the 4-8 LLM calls per ticket dominate cost and latency, and prompt size stays flat because tools return truncated top-k results. At 10x: tickets/accounts/usage in a database indexed on (account_id, created_at, status) with cached daily account health; KB in a search engine with hybrid BM25 + embeddings and reranking (templated articles make keyword collisions common), filtering by module/status first; top-k ≤ 8; cache policies and the static prompt; parallel processing under rate limits; a smaller model for easy tickets.

## 4. Limitations
- Quality depends on the model; weak models may need corrections or hit the fallback. Confidence is self-reported, uncalibrated.
- "Last 3 months" = latest month vs 3 months earlier; an already-passed renewal date is not "within 90 days".
- Keyword retrieval (crude stemming) has no semantic matching; vague tickets return noise; symptom similarity is an LLM judgement.
- Name-only matching accepts exact names only, otherwise asks for clarification.
- Guardrails encode today's policy text (a test detects drift); policy changes need a code change.
- Prompt-injection defences are not a guarantee. A real security ticket that hits the fallback needs human triage; review security, P1 and low-confidence decisions.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # plus `pip install anthropic` or `openai` if used
cp .env.example .env                   # set LLM_PROVIDER, LLM_MODEL, API key (never commit .env)
python scripts/validate_data.py
python -m unittest discover -s tests -t .   # tools + loop tests (scripted fake LLM, plumbing only)
python run.py --set review             # writes outputs/*.json + outputs/run_log.jsonl
python scripts/check_outputs.py        # expects 10/10 valid
```
