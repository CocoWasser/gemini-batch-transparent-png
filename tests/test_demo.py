"""Export actual RGBA examples without making generation requests."""

from unittest.mock import patch

from PIL import Image

from tests.helpers import OfflineCase, bt
from scripts.generate_demo import main as demo_main


class DemoExportTests(OfflineCase):
    def test_export_existing_pair_needs_no_key_or_api_calls(self):
        self.fixture()
        image_id = self.ids[0]
        bt.compose_transparent(self.source / "white" / f"{image_id}.png",
                               self.source / "black" / f"{image_id}.png",
                               self.source / "transparent" / f"{image_id}.png")
        destination = self.root / "public"
        with patch.dict("os.environ", {}, clear=True):
            demo_main(["--tsv", str(self.tsv), "--id", image_id,
                       "--from-output", str(self.source), "--publish-to", str(destination)])
        self.assertEqual({path.name for path in destination.iterdir()},
                         {"white.png", "black.png", "transparent.png", "comparison.jpg"})
        with Image.open(destination / "transparent.png") as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getchannel("A").getextrema(), (0, 255))
        with Image.open(destination / "comparison.jpg") as image:
            self.assertGreater(image.width, image.height * 2)

    def test_incomplete_demo_is_not_published(self):
        self.fixture()
        destination = self.root / "public"
        with self.assertRaises(FileNotFoundError):
            demo_main(["--tsv", str(self.tsv), "--id", self.ids[0],
                       "--from-output", str(self.source), "--publish-to", str(destination)])
        self.assertFalse(destination.exists())
