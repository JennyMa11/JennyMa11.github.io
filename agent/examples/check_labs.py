"""Behavior checks for the handbook's offline labs (standard library only).

Run: python3 agent/examples/check_labs.py
"""

import tempfile
import unittest
from pathlib import Path

import repair_lab
import retrieval_lab


class RepairChecks(unittest.TestCase):
    def test_all_scenario_outcomes(self):
        expected = {
            "happy": ("success", 5), "timeout": ("success", 6),
            "invalid": ("rejected", 1), "false_final": ("rejected", 1),
            "stale": ("rejected", 6), "budget": ("budget_exhausted", 3),
            "injection": ("success", 5),
        }
        for scenario, outcome in expected.items():
            with self.subTest(scenario=scenario):
                report = repair_lab.run(scenario)
                self.assertEqual((report["status"], report["steps"]), outcome)
                calls = [event["call_id"] for event in report["trace"] if "call_id" in event]
                self.assertEqual(len(calls), len(set(calls)))

    def test_real_test_feedback_and_current_version(self):
        report = repair_lab.run()
        initial = report["trace"][0]["result"]
        final = report["trace"][3]["result"]
        self.assertFalse(initial["passed"])
        self.assertEqual({c["name"] for c in initial["checks"] if not c["passed"]},
                         {"boundary", "member", "negative"})
        self.assertTrue(final["passed"])
        self.assertNotEqual(initial["source_hash"], final["source_hash"])
        self.assertTrue(report["trace"][-1]["accepted"])

    def test_limit_before_final_does_not_claim_success(self):
        self.assertEqual(repair_lab.run(max_steps=4)["status"], "budget_exhausted")
        self.assertEqual(repair_lab.run(max_steps=5)["status"], "success")

    def test_tool_contract_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "work"
            root.mkdir()
            private = base / "secret.txt"
            private.write_text("private sentinel")
            source = root / "pricing.py"
            source.write_text(repair_lab.BUGGY)
            tools = repair_lab.Tools(root, "happy")
            invalid = [
                ("unknown", {}), ([], {}), ("read_file", None), ("read_file", {}),
                ("read_file", {"path": "pricing.py", "extra": True}),
                ("read_file", {"path": "../secret.txt"}),
                ("read_file", {"path": str(private)}),
                ("replace_file", {"path": "pricing.py", "content": 123}),
                ("replace_file", {"path": "pricing.py", "content": "x" * 4097}),
            ]
            for name, arguments in invalid:
                with self.subTest(name=name, arguments=arguments):
                    self.assertEqual(tools.execute(name, arguments)["status"], "error")
                    self.assertEqual(source.read_text(), repair_lab.BUGGY)
            self.assertEqual(tools.execute("read_file", {"path": "pricing.py"})["status"], "ok")
            self.assertEqual(private.read_text(), "private sentinel")

    def test_symlink_does_not_bypass_allowed_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "work"
            root.mkdir()
            private = base / "secret.txt"
            private.write_text("private sentinel")
            (root / "pricing.py").symlink_to(private)
            tools = repair_lab.Tools(root, "happy")
            self.assertEqual(tools.execute("read_file", {"path": "pricing.py"})["status"], "error")


class RetrievalChecks(unittest.TestCase):
    def test_rankings_and_recall(self):
        self.assertEqual([d["id"] for d in retrieval_lab.retrieve("member shipping fee", 2)], ["D2", "D1"])
        self.assertEqual([d["id"] for d in retrieval_lab.retrieve("standard member shipping", 1)], ["D1"])
        self.assertAlmostEqual(retrieval_lab.evaluate(1)["mean_recall_answerable"], 5 / 6)
        self.assertEqual(retrieval_lab.evaluate(2)["mean_recall_answerable"], 1)

    def test_visibility_and_unanswerable_query(self):
        hits = retrieval_lab.retrieve("internal account threshold member shipping", 10)
        self.assertEqual({d["id"] for d in hits}, {"D1", "D2"})
        self.assertEqual(retrieval_lab.retrieve("refund deadline"), [])
        question = retrieval_lab.evaluate(2)["questions"][-1]
        self.assertIsNone(question["recall"])
        self.assertTrue(question["should_abstain"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
