"""Numerical alpha reconstruction and background checks on known image pairs."""

import base64
import io

from PIL import Image
import numpy as np

from tests.helpers import OfflineCase, bt, repair, pixels


class MattingTests(OfflineCase):
    def test_core_recovers_opaque_transparent_and_soft_edges(self):
        self.fixture()
        target = self.source / "transparent" / "asset-A_1.png"
        bt.compose_transparent(self.source / "white" / target.name,
                               self.source / "black" / target.name, target)
        with Image.open(target) as image:
            rgba = np.asarray(image)
            self.assertEqual(image.mode, "RGBA")
        np.testing.assert_array_equal(rgba[0, 0], [0, 0, 0, 0])
        np.testing.assert_array_equal(rgba[64, 64], [32, 144, 64, 255])
        self.assertLessEqual(abs(int(rgba[35, 35, 3]) - 128), 1)
        np.testing.assert_allclose(rgba[35, 35, :3], [32, 144, 64], atol=2)
        np.testing.assert_array_equal(rgba[52, 52], [255, 255, 255, 255])

    def test_backgrounds_reject_nonopaque_and_nonuniform_images(self):
        for stage, bg in (("white", 255), ("black", 0)):
            with self.subTest(stage=stage):
                bt.validate_background(Image.fromarray(pixels(bg)), stage)
                with self.assertRaises(ValueError):
                    bt.validate_background(Image.new("RGB", (128, 128), "gray"), stage)
                with self.assertRaisesRegex(ValueError, "opaque"):
                    bt.validate_background(Image.new("RGBA", (128, 128), (bg, bg, bg, 0)), stage)
        with self.assertRaises(ValueError):
            bt.validate_background(Image.new("RGB", (16, 16), "white"), "white")
        with self.assertRaises(ValueError):
            bt.validate_background(Image.new("RGB", (128, 128), "white"), "other")

    def test_mismatched_dimensions_are_rejected(self):
        self.fixture()
        black = self.source / "black" / "asset-A_1.png"
        Image.new("RGB", (100, 100), "black").save(black)
        with self.assertRaises(bt.ImageSizeMismatch):
            bt.compose_transparent(self.source / "white" / black.name, black,
                                   self.source / "transparent" / black.name)

    def test_repair_uses_measured_backgrounds(self):
        self.fixture()
        white = self.source / "white" / "asset-A_1.png"
        black = self.source / "black" / white.name
        Image.fromarray(pixels(245)).save(white)
        Image.fromarray(pixels(8)).save(black)
        target = self.root / "fixed.png"
        repair.recomposite(white, black, target)
        with Image.open(target) as image:
            rgba = np.asarray(image)
        self.assertEqual(int(rgba[0, 0, 3]), 0)
        self.assertLessEqual(abs(int(rgba[35, 35, 3]) - 128), 2)

    def test_repair_rejects_insufficient_background_contrast(self):
        self.fixture()
        white = self.source / "white" / "asset-A_1.png"
        with self.assertRaisesRegex(ValueError, "contrast"):
            repair.recomposite(white, white, self.root / "invalid.png")

    def test_returned_alpha_is_validated_instead_of_silently_discarded(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (128, 128), (255, 255, 255, 0)).save(buffer, format="PNG")
        payload = {"candidates": [{"content": {"parts": [{"inlineData": {
            "data": base64.b64encode(buffer.getvalue()).decode("ascii")
        }}]}}]}
        image = bt.extract_image(payload)
        self.assertEqual(image.mode, "RGBA")
        with self.assertRaisesRegex(ValueError, "opaque"):
            bt.validate_background(image, "white")

    def test_palette_transparency_is_rejected(self):
        image = Image.new("P", (128, 128))
        image.putpalette([255, 255, 255] * 256)
        image.info["transparency"] = 0
        with self.assertRaisesRegex(ValueError, "opaque"):
            bt.validate_background(image, "white")
