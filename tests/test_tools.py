"""Tests for the lookup tools on the REAL dataset. No LLM is involved.

Run:  python -m unittest tests.test_tools -v
"""
import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tam_agent.data import load_accounts, load_tickets, load_usage
from tam_agent.kb import load_kb
from tam_agent.retrieval import canonical_module, extract_error_code
from tam_agent.tools import Tools


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickets_df = load_tickets()
        cls.kb = load_kb()
        cls.T = Tools(load_accounts(), cls.tickets_df, load_usage(), cls.kb)
        cls.by_id = cls.tickets_df.set_index("ticket_id")

    def ticket(self, tid):
        return self.by_id.loc[tid].to_dict() | {"ticket_id": tid}

    # ---- policy thresholds used in tools.py really are in the policy text
    def test_policy_numbers_match_code_constants(self):
        text = " ".join(p["body"] for p in self.T.policies)
        self.assertEqual(len(self.T.policies), 3)
        for needle in ("within 90 days of renewal", "more than 25%", "2 or more open escalations", "previous 7 days"):
            self.assertIn(needle, text)

    # ---- module aliases
    def test_every_module_label_in_data_is_mapped_or_blank(self):
        unmapped = {m for m in self.tickets_df["module"].unique() if m and canonical_module(m) is None}
        self.assertEqual(unmapped, set())
        self.assertIsNone(canonical_module(""))

    def test_error_code_extracted_from_text(self):
        self.assertEqual(extract_error_code("Following up. Still shows E-6105. Thanks"), "E-6105")
        self.assertIsNone(extract_error_code("nothing here"))

    # ---- account resolution
    def test_blank_account_id_resolved_by_domain_and_lookalike_not_chosen(self):
        t = self.ticket("TKT-169327")
        self.assertEqual(t["account_id"], "")
        r = self.T.call("resolve_account", {"account_id": "", "requester_domain": t["requester_domain"],
                                            "company_name": t["company_name_raw"]}, t)
        self.assertEqual((r["status"], r["account_id"], r["method"]), ("resolved", "ACC-4850", "requester_domain"))
        self.assertTrue(any("ACC-3405" in x for x in r["lookalike_accounts_not_selected"]))

    def test_similar_name_alone_is_uncertain_not_resolved(self):
        r = self.T.call("resolve_account", {"company_name": "Ironleaf Foods EU"}, {})
        self.assertEqual(r["status"], "uncertain")
        self.assertIsNone(r["account_id"])

    def test_exact_name_with_legal_suffix_resolves(self):
        r = self.T.call("resolve_account", {"company_name": "Clearwater Industries Inc."}, {})
        self.assertEqual(r["account_id"], "ACC-1723")

    def test_conflicting_id_and_domain_is_uncertain(self):
        r = self.T.call("resolve_account", {"account_id": "ACC-3405", "requester_domain": "ironleaf-foods-europe.example"}, {})
        self.assertEqual(r["status"], "uncertain")

    def test_unknown_and_empty_inputs_unresolved(self):
        self.assertEqual(self.T.call("resolve_account", {"account_id": "ACC-0000", "requester_domain": "zzz.example"}, {})["status"], "unresolved")
        self.assertEqual(self.T.call("resolve_account", {}, {})["status"], "unresolved")

    # ---- account context
    def test_risk_override_computed_from_data(self):
        t = self.ticket("TKT-119877")
        ctx = self.T.call("get_account_context", {"account_id": "ACC-2431"}, t)
        self.assertEqual(ctx["days_to_renewal"], 61)                       # 2026-12-20 minus 2026-10-20
        self.assertEqual(ctx["open_escalations"], 2)
        self.assertLess(ctx["usage"]["active_users_change_3m_pct"], -25)
        self.assertTrue(ctx["risk_override"]["applies"])

    def test_risk_override_requires_renewal_window(self):
        t = self.ticket("TKT-169327")                                      # users down >25% but renewal far away
        ctx = self.T.call("get_account_context", {"account_id": "ACC-4850"}, t)
        self.assertTrue(ctx["risk_override"]["active_users_down_over_25pct"])
        self.assertFalse(ctx["risk_override"]["applies"])

    def test_sso_change_date_exposed(self):
        ctx = self.T.call("get_account_context", {"account_id": "ACC-2507"}, self.ticket("TKT-160094"))
        self.assertEqual(ctx["days_since_sso_change"], 6)

    # ---- duplicates
    def test_duplicate_candidate_found_even_when_error_code_only_in_description(self):
        t = self.ticket("TKT-192732")
        self.assertEqual(t["error_code"], "")
        r = self.T.call("find_related_tickets", {"account_id": "ACC-6031"}, t)
        top = r["candidates"][0]
        self.assertEqual(top["ticket_id"], "TKT-122022")
        self.assertTrue(top["meets_hard_duplicate_criteria"])
        self.assertEqual(r["compared_error_code"], "E-6105")

    def test_same_module_different_error_code_is_not_a_duplicate(self):
        r = self.T.call("find_related_tickets", {"account_id": "ACC-6031"}, self.ticket("TKT-192732"))
        other = [c for c in r["candidates"] if c["ticket_id"] == "TKT-152044"][0]   # Spend Analytics, E-6169
        self.assertFalse(other["meets_hard_duplicate_criteria"])
        self.assertIs(other["checks"]["same_error_code"], False)

    def test_current_ticket_and_future_tickets_excluded(self):
        t = self.ticket("TKT-192732")
        ids = [c["ticket_id"] for c in self.T.call("find_related_tickets", {"account_id": "ACC-6031"}, t)["candidates"]]
        self.assertNotIn("TKT-192732", ids)

    def test_blank_account_id_tickets_count_as_history(self):
        blank = self.tickets_df[self.tickets_df.account_id == ""].iloc[0]["ticket_id"]
        row = self.T.tickets[blank]
        self.assertTrue(row["eff_account"].startswith("ACC-"))
        self.assertIn(row, self.T.tickets_by_account[row["eff_account"]])

    # ---- knowledge base
    def test_error_code_search_ranks_exact_article_first(self):
        r = self.T.call("search_kb", {"query": "invoice OCR failing", "error_code": "E-4102"}, {})
        top2 = {x["id"] for x in r["results"][:2]}
        self.assertEqual(top2, {"KB-3985", "KB-7914"})
        self.assertEqual(r["results"][0]["status"] in ("active_incident", "current"), True)

    def test_deprecated_article_is_replaced_by_superseding_article(self):
        r = self.T.call("search_kb", {"query": "E-2207 e-signature rejected", "error_code": "E-2207"}, {})
        ids = [x["id"] for x in r["results"]]
        self.assertIn("KB-1786", ids)
        self.assertNotIn("KB-2605", ids)
        self.assertIn("deprecated", r["results"][0]["note"])
        read = self.T.call("read_kb_article", {"article_id": "KB-2605"}, {})
        self.assertEqual(read["id"], "KB-1786")

    def test_no_deprecated_article_is_ever_returned_by_search(self):
        deprecated = {a["id"] for a in self.kb if a["meta"]["status"] == "deprecated"}
        self.assertEqual(len(deprecated), 15)
        for a in self.kb:                                                   # use each article's title as a query
            if a["meta"]["id"] in deprecated:
                res = self.T.call("search_kb", {"query": a["meta"]["title"]}, {})
                self.assertFalse(deprecated & {x["id"] for x in res["results"]}, a["id"])

    def test_read_article_lists_related_links(self):
        r = self.T.call("read_kb_article", {"article_id": "KB-4795"}, {})
        self.assertIn("KB-7889", [x["id"] for x in r["related"]])

    def test_policies_returned_in_full(self):
        r = self.T.call("get_policies", {}, {})
        self.assertEqual({p["id"] for p in r["policies"]}, {"KB-2914", "KB-5930", "KB-6401"})

    # ---- argument validation
    def test_invalid_calls_return_errors_instead_of_raising(self):
        for name, args in [("nope", {}), ("search_kb", {}), ("search_kb", {"query": "x", "bogus": 1}),
                           ("get_account_context", {"account_id": "ACC-0000"}), ("read_kb_article", {"article_id": "KB-0"}),
                           ("search_kb", "not a dict")]:
            self.assertIn("error", self.T.call(name, args, self.ticket("TKT-169327")), (name, args))

    # ---- scale / robustness over the full ticket table
    def test_tools_survive_a_large_sample_of_real_tickets_quickly(self):
        ids = random.Random(1).sample(list(self.tickets_df["ticket_id"]), 400)
        start = time.time()
        for tid in ids:
            t = self.ticket(tid)
            r = self.T.call("resolve_account", {"account_id": t["account_id"], "requester_domain": t["requester_domain"],
                                                "company_name": t["company_name_raw"]}, t)
            self.assertEqual(r["status"], "resolved", tid)
            acc = r["account_id"]
            for name, args in [("get_account_context", {"account_id": acc}), ("find_related_tickets", {"account_id": acc}),
                               ("search_kb", {"query": t["subject"] + " " + t["description"], "module": t["module"],
                                              "error_code": t["error_code"]})]:
                self.assertNotIn("error", self.T.call(name, args, t), (tid, name))
        self.assertLess(time.time() - start, 30)

    def test_every_ticket_domain_resolves_to_the_account_in_its_own_id(self):
        wrong = 0
        for tid, t in self.T.tickets.items():
            if t["account_id"]:
                wrong += self.T.by_domain[t["requester_domain"].lower()] != t["account_id"]
        self.assertEqual(wrong, 0)


if __name__ == "__main__":
    unittest.main()
