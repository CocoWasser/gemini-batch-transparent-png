"""Interior artifacts, contact sheet pagination, and local-only review."""

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests.helpers import OfflineCase, bt
from review import assess_quality, checkerboard, write_review


class ReviewTests(OfflineCase):
    def compose(self, image_id="asset-A_1"):
        target = self.source / "transparent" / f"{image_id}.png"
        bt.compose_transparent(self.source / "white" / target.name,
                               self.source / "black" / target.name, target)
        return target

    def test_interior_white_patch_is_flagged_even_when_edges_pass(self):
        self.fixture()
        for stage in ("white", "black"):
            path = self.source / stage / "asset-A_1.png"
            with Image.open(path) as image:
                data = np.array(image)
            data[58:80, 58:80] = 255
            Image.fromarray(data).save(path)
            with Image.open(path) as image:
                bt.validate_background(image, stage)
        target = self.compose()
        quality = assess_quality(target, self.source / "white" / target.name,
                                  self.source / "black" / target.name)
        self.assertIn("near-white", quality["review_notes"])
        self.assertEqual(quality["nontransparent_edge_pct"], 0)
        self.assertTrue(target.exists())

    def test_changed_foreground_colors_are_flagged(self):
        self.fixture()
        path = self.source / "black" / "asset-A_1.png"
        with Image.open(path) as image:
            data = np.array(image)
        data[40:88, 40:88, 0] = 200
        Image.fromarray(data).save(path)
        target = self.compose()
        report = assess_quality(target, self.source / "white" / target.name, path)
        self.assertIn("pair differs", report["review_notes"])

    def test_checkerboard_shows_transparent_pixels_and_preserves_opaque_white(self):
        rgba = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        rgba.putpixel((32, 32), (255, 255, 255, 255))
        preview = checkerboard(rgba)
        self.assertEqual(preview.getpixel((32, 32)), (255, 255, 255))
        self.assertNotEqual(preview.getpixel((0, 0)), preview.getpixel((17, 0)))

    def test_pagination_reports_missing_files_and_removes_obsolete_pages(self):
        self.fixture()
        self.compose()
        ids = self.ids + [f"missing-{number}" for number in range(15)]
        pages = write_review(ids, self.source)
        self.assertEqual(len(pages), 2)
        for page in pages:
            with Image.open(page) as image:
                self.assertLessEqual(image.height, 1300)
        rows = list(csv.DictReader((self.source / "review" / "overview.tsv").read_text().splitlines(), delimiter="\t"))
        self.assertEqual(len(rows), 17)
        self.assertEqual(rows[-1]["status"], "pending")
        write_review([self.ids[0]], self.source)
        self.assertFalse(pages[1].exists())

    def test_corrupt_output_is_reported_without_losing_other_previews(self):
        self.fixture()
        self.compose()
        (self.source / "transparent" / f"{self.ids[1]}.png").write_bytes(b"invalid image")
        pages = write_review(self.ids, self.source)
        self.assertEqual(len(pages), 1)
        report = (self.source / "review" / "overview.tsv").read_text()
        self.assertIn("invalid", report)
        self.assertIn(self.ids[0], report)

    def test_review_only_does_not_require_credentials_or_create_a_client(self):
        self.fixture()
        self.compose()
        with patch.dict("os.environ", {}, clear=True), patch.object(bt.genai, "Client") as client:
            bt.main([str(self.tsv), "--out", str(self.source), "--review-only"])
        client.assert_not_called()
        self.assertTrue((self.source / "review" / "overview-001.jpg").exists())

    def test_review_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            write_review(["../secret"], self.source)
        with self.assertRaises(ValueError):
            write_review(self.ids, self.source, name="../secret")
