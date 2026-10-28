"""Offline resume validation, quarantine, local repair, and saved-job recovery."""

import copy
import csv
import io
import struct
import zlib
from unittest.mock import Mock, patch

from PIL import Image

from tests.helpers import OfflineCase, FakeGemini, bt, repair


class ResumeValidationTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.fixture()
        self.work = self.source / "_work"
        self.state = repair.read_state(self.source)
        for stage in ("white", "black"):
            self.state["attempts"][stage] = dict.fromkeys(self.ids, 3)
        bt.save_state(self.state, self.work)
        for image_id in self.ids:
            bt.compose_transparent(
                bt.image_path(self.source, "white", image_id),
                bt.image_path(self.source, "black", image_id),
                bt.image_path(self.source, "transparent", image_id),
            )
        self.target = bt.image_path(self.source, "transparent", self.ids[0])
        self.peer = bt.image_path(self.source, "transparent", self.ids[1])

    def run_batch(self, fake, attempts=3):
        self.mock_api(fake)
        bt.main([str(self.tsv), "--out", str(self.source),
                 "--model", "source-model", "--size", "2K",
                 "--max-attempts", str(attempts)])

    def report_rows(self):
        with (self.source / "failures.tsv").open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def invalid_outputs(self):
        valid = self.target.read_bytes()
        cases = {"empty": b"", "garbage": b"not an image",
                 "truncated PNG": valid[:len(valid) // 2]}
        for mode, fmt in (("RGB", "PNG"), ("P", "PNG"),
                          ("LA", "PNG"), ("RGB", "JPEG"), ("RGBA", "TIFF")):
            buffer = io.BytesIO()
            image = Image.new(mode, (128, 128))
            if mode == "P":
                image.info["transparency"] = 0
            image.save(buffer, format=fmt)
            cases[f"{mode} {fmt}"] = buffer.getvalue()
        return cases

    def test_rejects_unreadable_truncated_wrong_format_and_non_rgba_outputs(self):
        for label, data in self.invalid_outputs().items():
            with self.subTest(label=label):
                self.target.write_bytes(data)
                self.assertIsNotNone(bt.transparent_output_error(self.target))
        self.target.unlink()
        self.assertIsNotNone(bt.transparent_output_error(self.target))

    def test_decode_is_required_even_when_png_chunk_checks_pass(self):
        # Keep valid chunk CRCs but damage the compressed pixel stream.
        data = bytearray(self.target.read_bytes())
        pos = data.index(b"IDAT")
        length = struct.unpack(">I", data[pos - 4:pos])[0]
        data[pos + 4] = 0  # Invalid zlib compression header.
        crc = zlib.crc32(data[pos:pos + 4 + length])
        data[pos + 4 + length:pos + 8 + length] = struct.pack(">I", crc)
        self.target.write_bytes(data)
        with Image.open(self.target) as image:
            image.verify()
        self.assertIsNotNone(bt.transparent_output_error(self.target))

    def test_valid_rgba_outputs_survive_without_sources_or_quality_retries(self):
        # Both extremes can be quality warnings, but are structurally valid.
        for path, alpha in ((self.target, 255), (self.peer, 0)):
            Image.new("RGBA", (128, 128), (255, 255, 255, alpha)).save(path)
        originals = {p: p.read_bytes() for p in (self.target, self.peer)}
        for stage in ("white", "black"):
            for image_id in self.ids:
                bt.image_path(self.source, stage, image_id).unlink()
        fake = FakeGemini()
        self.run_batch(fake)
        self.assertEqual(fake.created, [])
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)
        self.assertEqual(repair.read_state(self.source), self.state)
        self.assertFalse((self.work / "rejected").exists())
        self.assertEqual(self.report_rows(), [])

    def test_resume_quarantines_and_locally_recovers_invalid_outputs_at_limit(self):
        peer = self.peer.read_bytes()
        sources = {p: p.read_bytes() for stage in ("white", "black")
                   for p in (self.source / stage).glob("*.png")}
        fake = FakeGemini()
        for index, (label, data) in enumerate(self.invalid_outputs().items(), 1):
            with self.subTest(label=label):
                self.target.write_bytes(data)
                self.run_batch(fake)
                self.assertEqual(fake.created, [])
                self.assertIsNone(bt.transparent_output_error(self.target))
                self.assertEqual(self.peer.read_bytes(), peer)
                self.assertEqual({p: p.read_bytes() for p in sources}, sources)
                self.assertEqual(repair.read_state(self.source), self.state)
                rejected = list((self.work / "rejected" / "transparent").glob("*.png"))
                self.assertEqual(len(rejected), index)
                self.assertIn(data, [p.read_bytes() for p in rejected])
                self.assertEqual(self.report_rows(), [])
                with (self.source / "review" / "overview.tsv").open() as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
                self.assertTrue(all(row["status"] != "invalid" for row in rows))

    def test_check_existing_quarantines_before_completion_selection(self):
        self.target.write_bytes(b"broken")
        bt.check_existing(self.records, self.source, self.state, self.work)
        self.assertFalse(self.target.exists())
        self.assertEqual(repair.read_state(self.source), self.state)
        bt.compose_ready(self.records, self.source, self.state, self.work)
        self.assertIsNone(bt.transparent_output_error(self.target))

    def test_compose_ready_also_handles_invalid_output_during_saved_job_harvest(self):
        self.target.write_bytes(b"broken")
        bt.compose_ready(self.records, self.source, self.state, self.work)
        self.assertIsNone(bt.transparent_output_error(self.target))
        rejected = list((self.work / "rejected" / "transparent").glob("*.png"))
        self.assertEqual([p.read_bytes() for p in rejected], [b"broken"])

    def test_report_does_not_count_invalid_output_or_mutate_pending_job(self):
        self.target.write_bytes(b"broken")
        self.state["current"] = {"stage": "black", "status": "uncertain", "keys": self.ids}
        bt.save_state(self.state, self.work)
        original_state = copy.deepcopy(self.state)
        bt.write_report(self.records, self.source, self.state, 3)
        rows = self.report_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ID"], self.ids[0])
        self.assertEqual(rows[0]["Stage"], "compose")
        self.assertIn("Invalid transparent output", rows[0]["Error"])
        self.assertIn("Completed: 1", self.output.getvalue())
        self.assertEqual(self.target.read_bytes(), b"broken")
        self.assertEqual(self.state, original_state)
        self.assertEqual(repair.read_state(self.source), original_state)

    def test_selection_does_not_skip_invalid_final_with_missing_source(self):
        self.target.write_bytes(b"broken")
        for missing, expected_stage in (("black", "black"), ("white", "white")):
            with self.subTest(missing=missing):
                path = bt.image_path(self.source, missing, self.ids[0])
                held = path.read_bytes()
                path.unlink()
                self.state["attempts"][missing][self.ids[0]] = 2
                stage, rows = bt.select_next_batch(self.records, self.source, self.state, 100, 3)
                self.assertEqual(stage, expected_stage)
                self.assertEqual(rows, [self.records[0]])
                self.assertEqual(self.target.read_bytes(), b"broken")
                self.state["attempts"][missing][self.ids[0]] = 3
                self.assertEqual(bt.select_next_batch(self.records, self.source, self.state, 100, 3),
                                 (None, []))
                path.write_bytes(held)

    def test_missing_source_respects_limit_then_retries_only_unfinished_id(self):
        self.target.write_bytes(b"broken")
        bt.image_path(self.source, "black", self.ids[0]).unlink()
        peer = self.peer.read_bytes()
        fake = FakeGemini()
        self.run_batch(fake)
        self.assertEqual(fake.created, [])
        self.assertFalse(self.target.exists())
        rows = self.report_rows()
        self.assertEqual([(r["ID"], r["Stage"], r["Attempts"], r["Status"]) for r in rows],
                         [(self.ids[0], "black", "3", "Attempt limit reached")])
        self.run_batch(fake, attempts=4)
        self.assertEqual(len(fake.created), 1)
        self.assertTrue(fake.created[0][1])
        self.assertEqual([row["key"] for row in fake.created[0][2]], [self.ids[0]])
        self.assertIsNone(bt.transparent_output_error(self.target))
        self.assertEqual(self.peer.read_bytes(), peer)
        state = repair.read_state(self.source)
        self.assertEqual(state["attempts"]["white"], self.state["attempts"]["white"])
        self.assertEqual(state["attempts"]["black"], {self.ids[0]: 4, self.ids[1]: 3})
        self.assertEqual(self.report_rows(), [])

    def test_invalid_source_is_quarantined_without_resetting_attempts(self):
        self.target.write_bytes(b"broken output")
        source = bt.image_path(self.source, "black", self.ids[0])
        source.write_bytes(b"broken source")
        fake = FakeGemini()
        self.run_batch(fake)
        self.assertEqual(fake.created, [])
        self.assertFalse(self.target.exists())
        self.assertFalse(source.exists())
        rejected = list((self.work / "rejected" / "black").glob("*.png"))
        self.assertEqual([p.read_bytes() for p in rejected], [b"broken source"])
        self.assertEqual(repair.read_state(self.source)["attempts"], self.state["attempts"])
        self.assertEqual(self.report_rows()[0]["Status"], "Attempt limit reached")

    def test_local_composition_interruption_recovers_on_next_run_without_api(self):
        self.target.write_bytes(b"broken")
        fake = FakeGemini()
        replace = bt.os.replace

        def interrupt_final(source, destination):
            if destination == self.target:
                raise KeyboardInterrupt
            return replace(source, destination)

        with patch.object(bt.os, "replace", side_effect=interrupt_final):
            self.run_batch(fake)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.target.with_name(self.target.stem + ".tmp.png").exists())
        self.assertEqual(fake.created, [])
        self.run_batch(fake)
        self.assertIsNone(bt.transparent_output_error(self.target))
        self.assertEqual(fake.created, [])
        self.assertEqual(repair.read_state(self.source), self.state)

    def test_saved_black_job_is_harvested_once_before_repair_or_new_submission(self):
        self.target.write_bytes(b"broken")
        bt.image_path(self.source, "black", self.ids[0]).unlink()
        self.state["attempts"]["black"][self.ids[0]] = 2
        bt.save_state(self.state, self.work)
        fake = FakeGemini()
        get = fake.batches.get
        fake.batches.get = Mock(side_effect=KeyboardInterrupt)
        self.run_batch(fake)
        state = repair.read_state(self.source)
        self.assertEqual(len(fake.created), 1)
        self.assertEqual(state["current"]["status"], "submitted")
        self.assertEqual(state["attempts"]["black"][self.ids[0]], 3)
        # A corrupt final can reappear while a paid job is pending.
        self.target.write_bytes(b"broken again")
        fake.batches.get = get
        self.run_batch(fake)
        state = repair.read_state(self.source)
        self.assertEqual(len(fake.created), 1)
        self.assertIsNone(state["current"])
        self.assertEqual(state["attempts"]["black"][self.ids[0]], 3)
        self.assertIsNone(bt.transparent_output_error(self.target))
        self.assertEqual(self.report_rows(), [])

    def test_uncertain_job_blocks_new_requests_and_reports_corrupt_final(self):
        self.target.write_bytes(b"broken")
        bt.image_path(self.source, "black", self.ids[0]).unlink()
        self.state["attempts"]["black"][self.ids[0]] = 2
        bt.save_state(self.state, self.work)
        fake = FakeGemini(fail_after_create=True)
        self.run_batch(fake)
        self.target.write_bytes(b"broken again")
        before = repair.read_state(self.source)
        fake.batches.list = Mock(return_value=[])
        self.run_batch(fake)
        self.assertEqual(len(fake.created), 1)
        self.assertEqual(repair.read_state(self.source), before)
        self.assertIn("Invalid transparent output", self.report_rows()[0]["Error"])
        fake.batches.list = lambda: list(fake.jobs.values())
        self.run_batch(fake)
        self.assertEqual(len(fake.created), 1)
        state = repair.read_state(self.source)
        self.assertIsNone(state["current"])
        self.assertEqual(state["attempts"]["black"][self.ids[0]], 3)
        self.assertIsNone(bt.transparent_output_error(self.target))
