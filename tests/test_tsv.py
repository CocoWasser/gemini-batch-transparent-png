"""Input parsing and validation without network calls."""

from tests.helpers import OfflineCase, bt


class TSVTests(OfflineCase):
    def load(self, text, limit=None):
        self.tsv.write_text(text, encoding="utf-8")
        return bt.load_tsv(self.tsv, limit)

    def test_bom_blank_lines_and_header(self):
        rows = self.load("\ufeff\n\nID\tName\tPrompt\n\na-1\tRobot\tDraw a robot\n")
        self.assertEqual(rows, [("a-1", "Robot", "Draw a robot")])

    def test_headerless_unicode_names_and_quoted_multiline_prompts(self):
        rows = self.load('a_1\tCaf\u00e9 cup\t"First line\nSecond line"\n')
        self.assertEqual(rows, [("a_1", "Caf\u00e9 cup", "First line\nSecond line")])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            self.load("a\tA\tPrompt\na\tB\tOther\n")

    def test_invalid_ids_cannot_escape_the_output_directory(self):
        for image_id in ("../secret", "/absolute", "a/b", "a.png", "a b", ""):
            with self.subTest(image_id=image_id), self.assertRaisesRegex(ValueError, "Invalid ID"):
                self.load(f"{image_id}\tName\tPrompt\n")

    def test_empty_and_short_rows_and_empty_prompts_are_rejected(self):
        for text in ("", "\n", "ID\tName\tPrompt\n", "a\tName\n", "a\tName\t   \n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_limit_is_positive_and_preserves_file_order(self):
        rows = self.load("b\tB\tPrompt\na\tA\tPrompt\n", limit=1)
        self.assertEqual(rows[0][0], "b")
        for limit in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive"):
                bt.load_tsv(self.tsv, limit)

    def test_white_prompt_replaces_requested_transparency(self):
        prompt = bt.white_prompt("A toy ISOLATED ON A TRANSPARENT BACKGROUND")
        self.assertNotIn("transparent background", prompt.lower())
        self.assertIn("pure white", prompt.lower())
