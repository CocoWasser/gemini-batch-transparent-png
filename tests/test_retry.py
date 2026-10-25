"""Failed requests, quota pauses, saved jobs, and bounded retries."""

import json
from unittest.mock import Mock

from tests.helpers import OfflineCase, FakeGemini, bt, repair


class RetryTests(OfflineCase):
    def run_batch(self, fake, attempts=3):
        self.mock_api(fake)
        bt.main([str(self.tsv), "--out", str(self.source), "--max-attempts", str(attempts)])

    def test_failed_item_retries_without_regenerating_completed_peers(self):
        fake = FakeGemini(errors={("black", self.ids[0]): [500]})
        self.run_batch(fake)
        black_jobs = [job[2] for job in fake.created if job[1]]
        self.assertEqual([[row["key"] for row in job] for job in black_jobs],
                         [self.ids, [self.ids[0]]])
        state = repair.read_state(self.source)
        self.assertEqual(state["attempts"]["black"], {self.ids[0]: 2, self.ids[1]: 1})
        self.assertEqual(state["errors"]["black"], {})
        self.assertEqual(len(list((self.source / "review").glob("batch-*.jpg"))), 2)

    def test_persistent_failure_stops_at_the_attempt_limit(self):
        fake = FakeGemini(errors={("white", self.ids[0]): [500] * 10})
        self.run_batch(fake, attempts=2)
        state = repair.read_state(self.source)
        self.assertEqual(state["attempts"]["white"][self.ids[0]], 2)
        self.assertFalse((self.source / "transparent" / f"{self.ids[0]}.png").exists())
        self.assertTrue((self.source / "transparent" / f"{self.ids[1]}.png").exists())
        self.assertIn("Attempt limit reached", (self.source / "failures.tsv").read_text())
        count = len(fake.created)
        self.run_batch(fake, attempts=2)
        self.assertEqual(len(fake.created), count)
        self.run_batch(fake, attempts=3)
        self.assertEqual(repair.read_state(self.source)["attempts"]["white"][self.ids[0]], 3)

    def test_quota_failure_saves_successes_and_waits_for_next_run(self):
        fake = FakeGemini(errors={("white", self.ids[0]): [429]})
        self.run_batch(fake)
        self.assertEqual(len(fake.created), 1)
        self.assertTrue((self.source / "white" / f"{self.ids[1]}.png").exists())
        self.assertIsNone(repair.read_state(self.source)["current"])
        self.run_batch(fake)
        self.assertTrue(all((self.source / "transparent" / f"{image_id}.png").exists()
                            for image_id in self.ids))

    def test_unknown_submission_is_not_retried_without_confirmation(self):
        fake = FakeGemini(fail_after_create=True)
        self.run_batch(fake)
        before = len(fake.created)
        fake.batches.list = Mock(return_value=[])
        self.run_batch(fake)
        self.assertEqual(len(fake.created), before)
        self.assertEqual(repair.read_state(self.source)["current"]["status"], "uncertain")
        self.assertIn("--confirm-not-created", self.output.getvalue())

    def test_attempts_are_counted_once_after_polling_interruption(self):
        fake = FakeGemini()
        original_get = fake.get
        fake.batches.get = Mock(side_effect=ConnectionError("Synthetic polling interruption"))
        self.run_batch(fake)
        state = repair.read_state(self.source)
        self.assertEqual(state["attempts"]["white"], dict.fromkeys(self.ids, 1))
        self.assertEqual(state["current"]["status"], "submitted")
        fake.batches.get = original_get
        self.run_batch(fake)
        self.assertEqual(repair.read_state(self.source)["attempts"]["white"], dict.fromkeys(self.ids, 1))
        self.assertEqual(len(fake.created), 2)

    def test_legacy_saved_job_is_recovered_before_new_submission(self):
        self.fixture()
        work = self.source / "_work"
        state = repair.read_state(self.source)
        state["current"] = None
        state["jobs"] = {"black": "batches/existing-job"}
        (work / "black_requests.jsonl").write_text(json.dumps({"key": self.ids[0]}) + "\n")
        bt.save_state(state, work)
        recovered = bt.initialize_state(self.records, work, "source-model", "2K")
        self.assertEqual(recovered["current"]["job_name"], "batches/existing-job")
        self.assertEqual(recovered["current"]["keys"], [self.ids[0]])
        self.assertEqual(recovered["jobs"], {})
