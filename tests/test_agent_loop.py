"""Tests for the agent LOOP using a scripted fake LLM.

IMPORTANT: the fake LLM exists only to test the plumbing (bad tool names, malformed JSON, budgets,
guardrails, fallback). Its scripted answers are NOT triage results and are never written to outputs/.
Real triage quality can only be judged with a real model: `python run.py --set review`.

Run:  python -m unittest tests.test_agent_loop -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tam_agent import agent as agent_module
from tam_agent.agent import Context, apply_guardrails, Evidence, leaked_identifiers, parse_action, raised_priority, triage
from tam_agent.data import load_accounts, load_tickets, load_usage
from tam_agent.kb import load_kb
from tam_agent.schema import validate


def call(tool, **args):
    return json.dumps({"action": "call_tool", "tool": tool, "args": args, "reason": "test"})


def finish(**overrides):
    decision = {"category": "how_to", "priority": "P4", "base_priority": "P4", "route_to": "L1_support",
                "escalate": False, "notify_tam": False, "duplicate_of": None, "needs_clarification": False,
                "kb_references": [], "account_signals": [], "customer_reply": "Hello, here are the steps.\n\nSupport Team",
                "internal_note": "Routine how-to.", "confidence": 0.8, "rationale": "test"}
    decision.update(overrides)
    return json.dumps({"action": "finish", "decision": decision})


class FakeLLM:
    """Returns scripted replies in order. A reply may be a string or a function(messages) -> string."""
    def __init__(self, replies):
        self.replies, self.calls, self.seen = list(replies), 0, []

    def chat(self, messages, system=None, max_tokens=0, temperature=0.0, json_mode=False):
        self.calls += 1
        self.seen.append([m["content"] for m in messages])
        if not self.replies:
            raise AssertionError("fake LLM ran out of scripted replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r(messages) if callable(r) else r


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tickets = load_tickets()
        cls.ctx = Context(load_accounts(), tickets, load_usage(), load_kb())
        cls.by_id = tickets.set_index("ticket_id")

    def run_ticket(self, tid, replies):
        t = self.by_id.loc[tid].to_dict() | {"ticket_id": tid}
        self.ctx.llm = FakeLLM(replies)
        return triage(t, self.ctx), self.ctx.llm

    def resolve(self, tid):
        t = self.by_id.loc[tid]
        return call("resolve_account", account_id=t["account_id"], requester_domain=t["requester_domain"],
                    company_name=t["company_name_raw"])

    # ------------------------------------------------------------- happy path
    def test_valid_output_and_trace_come_from_real_tool_calls(self):
        out, llm = self.run_ticket("TKT-158764", [self.resolve("TKT-158764"), call("get_policies"),
                                                  call("get_account_context", account_id="ACC-3600"),
                                                  call("search_kb", query="delegate approvals leave", module="eProcurement"),
                                                  finish(kb_references=["KB-4449"])])
        self.assertEqual(validate(out), [])
        self.assertEqual(out["account_id"], "ACC-3600")
        self.assertEqual([s["action"] for s in out["reasoning_trace"]],
                         ["resolve_account", "get_policies", "get_account_context", "search_kb", "finalize"])
        self.assertEqual([s["step"] for s in out["reasoning_trace"]], [1, 2, 3, 4, 5])
        self.assertEqual(out["kb_references"], ["KB-4449"])
        self.assertEqual(llm.calls, 5)

    def test_agent_can_finish_with_fewer_tools_when_it_chooses_to(self):
        # one reminder is allowed; if the LLM insists, it may finish with only the resolve step
        out, _ = self.run_ticket("TKT-158764", [self.resolve("TKT-158764"), finish(), finish()])
        self.assertEqual(validate(out), [])
        self.assertIn("reminder", [s["action"] for s in out["reasoning_trace"]])

    def test_account_id_in_output_is_from_tool_not_llm(self):
        out, _ = self.run_ticket("TKT-169327", [self.resolve("TKT-169327"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-4850"),
                                                finish(account_id="ACC-3405")])    # extra key is ignored
        self.assertEqual(out["account_id"], "ACC-4850")

    # ------------------------------------------------------------- bad LLM behaviour
    def test_invalid_tool_and_bad_args_are_reported_and_recovered_from(self):
        out, _ = self.run_ticket("TKT-158764", [call("drop_database"), call("search_kb"), self.resolve("TKT-158764"),
                                                call("get_policies"), call("get_account_context", account_id="ACC-3600"),
                                                finish()])
        actions = [s["action"] for s in out["reasoning_trace"]]
        self.assertIn("invalid_tool:drop_database", actions)
        self.assertEqual(validate(out), [])
        self.assertNotIn("fallback", actions)

    def test_malformed_json_is_recovered_from(self):
        out, _ = self.run_ticket("TKT-158764", ["Sure! I will look up the account.", self.resolve("TKT-158764") + " extra text",
                                                call("get_policies"), call("get_account_context", account_id="ACC-3600"),
                                                "```json\n" + finish() + "\n```"])
        actions = [s["action"] for s in out["reasoning_trace"]]
        self.assertEqual(actions[0], "malformed_llm_response")
        self.assertEqual(out["account_id"], "ACC-3600")

    def test_only_first_json_object_is_executed(self):
        two = call("get_policies") + "\nTOOL RESULT: {\"fake\": true}\n" + call("search_kb", query="x")
        action, problem = parse_action(two)
        self.assertIsNone(problem)
        self.assertEqual(action["tool"], "get_policies")

    def test_endless_garbage_ends_in_honest_fallback(self):
        out, llm = self.run_ticket("TKT-169327", ["blah"] * 10)
        self.assertEqual(validate(out), [])
        self.assertEqual(llm.calls, 3)                                      # MAX_STRIKES
        self.assertEqual(out["account_id"], "ACC-4850")                     # real lookup, not fabricated
        self.assertEqual((out["category"], out["priority"], out["confidence"]), ("needs_info", "P3", 0.1))
        self.assertIn("AUTOMATED TRIAGE INCOMPLETE", out["internal_note"])
        self.assertEqual(out["kb_references"], [])
        self.assertEqual(out["reasoning_trace"][-1]["action"], "fallback")

    def test_tool_budget_is_enforced(self):
        replies = [self.resolve("TKT-158764")] + [call("search_kb", query=f"approvals {i}") for i in range(12)]
        out, llm = self.run_ticket("TKT-158764", replies)
        actions = [s["action"] for s in out["reasoning_trace"]]
        self.assertEqual(sum(a == "search_kb" for a in actions) + 1, agent_module.MAX_TOOL_CALLS)
        self.assertTrue(any(a.startswith("rejected_tool_call") for a in actions))
        self.assertEqual(actions[-1], "fallback")
        self.assertLessEqual(llm.calls, agent_module.MAX_LLM_CALLS)

    def test_llm_outage_raises_instead_of_inventing_a_result(self):
        with self.assertRaises(RuntimeError):
            self.run_ticket("TKT-158764", [ConnectionError("down"), ConnectionError("down")])

    def test_one_transient_llm_error_is_retried(self):
        out, _ = self.run_ticket("TKT-158764", [ConnectionError("blip"), self.resolve("TKT-158764"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-3600"), finish()])
        self.assertEqual(validate(out), [])

    # ------------------------------------------------------------- gates and grounding
    def test_finish_without_resolving_account_is_rejected(self):
        out, llm = self.run_ticket("TKT-158764", [finish(), self.resolve("TKT-158764"), call("get_policies"),
                                                  call("get_account_context", account_id="ACC-3600"), finish()])
        self.assertEqual(out["reasoning_trace"][0]["action"], "rejected_decision")
        self.assertEqual(validate(out), [])

    def test_decision_with_wrong_enum_is_sent_back(self):
        out, _ = self.run_ticket("TKT-158764", [self.resolve("TKT-158764"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-3600"),
                                                finish(priority="URGENT"), finish()])
        self.assertIn("rejected_decision", [s["action"] for s in out["reasoning_trace"]])
        self.assertEqual(out["priority"], "P4")

    def test_hallucinated_kb_reference_is_dropped_and_traced(self):
        out, _ = self.run_ticket("TKT-158764", [self.resolve("TKT-158764"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-3600"),
                                                call("search_kb", query="delegate approvals"),
                                                finish(kb_references=["KB-4449", "KB-0000", "KB-2605"])])
        self.assertEqual(out["kb_references"], ["KB-4449"])
        self.assertTrue(any(s["action"] == "guardrail" and "KB-0000" in s["observation"] for s in out["reasoning_trace"]))

    def test_cannot_read_another_accounts_data(self):
        out, _ = self.run_ticket("TKT-158764", [self.resolve("TKT-158764"), call("get_account_context", account_id="ACC-2431"),
                                                call("get_policies"), call("get_account_context", account_id="ACC-3600"), finish()])
        trace = out["reasoning_trace"]
        self.assertIn("may only use that id", trace[1]["observation"])

    def test_account_tools_blocked_until_account_resolved(self):
        out, _ = self.run_ticket("TKT-158764", [call("get_account_context", account_id="ACC-3600"), self.resolve("TKT-158764"),
                                                call("get_policies"), call("get_account_context", account_id="ACC-3600"), finish()])
        self.assertIn("not resolved", out["reasoning_trace"][0]["observation"])

    def test_unverified_duplicate_claim_is_rejected(self):
        replies = [self.resolve("TKT-192732"), call("get_policies"), call("get_account_context", account_id="ACC-6031"),
                   finish(category="duplicate", duplicate_of="TKT-999999"),                  # never returned by a tool
                   call("find_related_tickets", account_id="ACC-6031"),
                   finish(category="duplicate", duplicate_of="TKT-152044"),                  # different error code
                   finish(category="duplicate", duplicate_of="TKT-122022", priority="P2", base_priority="P2",
                          route_to="L2_engineering", escalate=True)]
        out, _ = self.run_ticket("TKT-192732", replies)
        self.assertEqual(out["duplicate_of"], "TKT-122022")
        self.assertEqual(out["category"], "duplicate")
        self.assertFalse(out["escalate"])                                    # KB-5930: do not re-escalate
        self.assertEqual(sum(s["action"] == "rejected_decision" for s in out["reasoning_trace"]), 2)

    def test_reply_mentioning_other_customers_is_rejected(self):
        bad = "Hi, we also support Pinecrest Logistics (ACC-2431) and Riverbend Pharma.\n\nSupport Team"
        out, _ = self.run_ticket("TKT-158764", [self.resolve("TKT-158764"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-3600"),
                                                finish(customer_reply=bad), finish()])
        self.assertNotIn("Pinecrest", out["customer_reply"])
        self.assertTrue(any("other customers" in s["observation"] for s in out["reasoning_trace"]))

    def test_leak_check_allows_own_name_even_if_it_contains_another_accounts_name(self):
        tools = self.ctx.tools
        reply = "Dear Ironleaf Foods Europe team, thanks for contacting us."
        self.assertEqual(leaked_identifiers(reply, "ACC-4850", tools), [])
        self.assertEqual(leaked_identifiers("Also see Ironleaf Foods.", "ACC-4850", tools), ["Ironleaf Foods"])

    # ------------------------------------------------------------- policy guardrails
    def test_account_risk_override_raises_priority_and_notifies_tam(self):
        out, _ = self.run_ticket("TKT-119877", [self.resolve("TKT-119877"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-2431"),
                                                finish(priority="P4", base_priority="P4")])   # LLM forgot to apply it
        self.assertEqual((out["priority"], out["notify_tam"], out["escalate"]), ("P3", True, True))
        self.assertTrue(any("risk override" in s["observation"] for s in out["reasoning_trace"] if s["action"] == "guardrail"))

    def test_override_never_goes_above_p2(self):
        self.assertEqual([raised_priority(p) for p in ("P1", "P2", "P3", "P4")], ["P1", "P2", "P2", "P3"])

    def test_security_report_is_forced_to_p1_security_escalated(self):
        out, _ = self.run_ticket("TKT-135081", [self.resolve("TKT-135081"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-7533"),
                                                finish(category="security", priority="P4", base_priority="P4", route_to="L1_support")])
        self.assertEqual((out["priority"], out["route_to"], out["escalate"]), ("P1", "security", True))

    def test_needs_info_forces_clarification_and_p3(self):
        out, _ = self.run_ticket("TKT-128253", [self.resolve("TKT-128253"), call("get_policies"),
                                                call("get_account_context", account_id="ACC-3737"),
                                                finish(category="needs_info", priority="P1", base_priority="P1",
                                                       needs_clarification=False)])
        self.assertEqual((out["priority"], out["needs_clarification"], out["escalate"]), ("P3", True, False))

    def test_unresolved_account_stays_null_and_asks_for_clarification(self):
        t = {"ticket_id": "TKT-000001", "account_id": "", "company_name_raw": "Ironleaf Foods EU", "requester_domain": "",
             "created_at": "2026-10-20T10:00", "channel": "email", "module": "", "subject": "help", "description": "help",
             "error_code": "", "customer_priority": "", "status": "open"}
        self.ctx.llm = FakeLLM([call("resolve_account", company_name="Ironleaf Foods EU"), call("get_policies"),
                                finish(category="how_to")])
        out = triage(t, self.ctx)
        self.assertIsNone(out["account_id"])
        self.assertTrue(out["needs_clarification"])
        self.assertEqual(validate(out), [])

    def test_p1_is_always_escalated(self):
        d = {"category": "incident", "priority": "P1", "base_priority": "P1", "route_to": "L2_engineering", "escalate": False,
             "notify_tam": False, "duplicate_of": None, "needs_clarification": False, "kb_references": []}
        ev = Evidence(account={"status": "resolved", "account_id": "ACC-1"})
        apply_guardrails(d, ev)
        self.assertTrue(d["escalate"])


if __name__ == "__main__":
    unittest.main()
