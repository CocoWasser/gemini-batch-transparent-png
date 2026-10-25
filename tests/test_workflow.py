"""End-to-end mocked Batch execution, recovery, and repair workflows."""

import contextlib
import csv
import io
import os
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests.helpers import OfflineCase, FakeGemini, PROJECT, bt, repair


class WorkflowTests(OfflineCase):
    def test_example_has_four_valid_distinct_prompts(self):
        self.assertEqual(len(bt.load_tsv(PROJECT / 'example.tsv')), 4)

    def test_arbitrary_ids_and_path_traversal_rejection(self):
        self.assertEqual(repair.parse_ids(self.ids + self.ids), self.ids)
        for bad in ['../private', 'x/y', '', '\u542b\u4e2d\u6587', 'a.png']:
            with self.assertRaises(ValueError):
                repair.parse_ids([bad])

    def test_local_recovers_alpha_and_keeps_white_foreground(self):
        self.fixture()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.cli('local', *self.ids), 0)
        arr = np.asarray(Image.open(self.rep / 'local' / f'{self.ids[0]}.png'))
        self.assertEqual(arr.shape, (128, 128, 4))
        self.assertEqual(int(arr[0, 0, 3]), 0)
        self.assertEqual(int(arr[64, 64, 3]), 255)
        self.assertLessEqual(abs(int(arr[35, 35, 3])-128), 1)
        np.testing.assert_array_equal(arr[52, 52], [255, 255, 255, 255])
        self.assertTrue((self.rep / 'local_preview' / f'{self.ids[0]}.jpg').exists())
        self.assertFalse((self.source / 'transparent' / f'{self.ids[0]}.png').exists())

    def test_accept_backs_up_original_and_prevalidates_all(self):
        self.fixture()
        self.cli('local', *self.ids)
        original = self.source / 'transparent' / f'{self.ids[0]}.png'
        original.write_bytes(b'original-data')
        missing = self.rep / 'local' / f'{self.ids[1]}.png'
        held = missing.read_bytes()
        missing.unlink()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli('accept', 'local', *self.ids)
        self.assertEqual(original.read_bytes(), b'original-data')
        missing.write_bytes(held)
        self.assertEqual(self.cli('accept', 'local', *self.ids), 0)
        backups = list((self.source / '_work' / 'quality_backup').glob(f'*/{self.ids[0]}.png'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b'original-data')
        self.assertEqual(original.read_bytes(), (self.rep / 'local' / original.name).read_bytes())

    def test_core_white_black_flow_and_resume_without_resubmission(self):
        fake = FakeGemini()
        self.mock_api(fake)
        args = [str(self.tsv), '--out', str(self.source), '--batch-size', '2']
        bt.main(args)
        self.assertEqual([job[1] for job in fake.created], [False, True])
        self.assertEqual(len(list((self.source / 'transparent').glob('*.png'))), 2)
        self.assertTrue(fake.closed)
        state = repair.read_state(self.source)
        self.assertIsNone(state['current'])
        report_rows = list(csv.reader((self.source / 'failures.tsv').read_text().splitlines(), delimiter='\t'))
        self.assertEqual(len(report_rows), 1)
        bt.main(args)
        self.assertEqual(len(fake.created), 2)
        changed = self.tsv.read_text().replace('An illustration', 'Changed illustration')
        self.tsv.write_text(changed)
        with self.assertRaises(ValueError):
            bt.main(args)
        with bt.output_lock(self.source):
            pass

    def test_uncertain_submission_is_recovered_without_duplicate(self):
        fake = FakeGemini(fail_after_create=True)
        self.mock_api(fake)
        args = [str(self.tsv), '--out', str(self.source)]
        bt.main(args)
        self.assertEqual(repair.read_state(self.source)['current']['status'], 'uncertain')
        self.assertEqual(len(fake.created), 1)
        bt.main(args)
        self.assertEqual(len(fake.created), 2)
        self.assertEqual(len(list((self.source / 'transparent').glob('*.png'))), 2)

    def test_regen_inherits_settings_and_retry_preserves_rejected_files(self):
        self.fixture()
        fake = FakeGemini()
        self.mock_api(fake)
        white = (self.source / 'white' / f'{self.ids[0]}.png').read_bytes()
        self.assertEqual(self.cli('regen', self.ids[0], '--tsv', str(self.tsv)), 0)
        self.assertEqual(len(fake.created), 1)
        model, is_black, rows = fake.created[0]
        self.assertEqual(model, 'source-model')
        self.assertTrue(is_black)
        request = rows[0]['request']
        self.assertEqual(request['generationConfig']['imageConfig']['imageSize'], '2K')
        self.assertEqual(request['contents'][0]['parts'][0]['text'], repair.STRICT_BLACK_PROMPT)
        self.assertEqual((self.rep / 'white' / f'{self.ids[0]}.png').read_bytes(), white)
        self.assertEqual(self.cli('regen', self.ids[0], '--tsv', str(self.tsv)), 0)
        self.assertEqual(len(fake.created), 1)
        old_black = (self.rep / 'black' / f'{self.ids[0]}.png').read_bytes()
        self.assertEqual(self.cli('retry', self.ids[0]), 0)
        backups = list((self.rep / '_work' / 'manual_rejections').glob(f'*/black/{self.ids[0]}.png'))
        self.assertEqual(backups[0].read_bytes(), old_black)
        self.assertFalse((self.rep / 'black' / f'{self.ids[0]}.png').exists())
        self.assertEqual(self.cli('regen', self.ids[0], '--tsv', str(self.tsv)), 0)
        self.assertEqual(len(fake.created), 2)
        self.assertEqual(self.cli('accept', 'regen', self.ids[0]), 0)

    def test_pending_batch_cannot_be_rejected_or_accepted(self):
        self.fixture()
        self.cli('local', *self.ids)
        self.rep.joinpath('_work').mkdir(exist_ok=True)
        state = bt.initialize_state(self.records, self.rep / '_work', 'source-model', '2K')
        state['current'] = {'keys': self.ids, 'status': 'uncertain'}
        bt.save_state(state, self.rep / '_work')
        original = (self.rep / '_work' / 'state.json').read_bytes()
        for command, args in [('retry', [self.ids[0]]), ('accept', ['local', self.ids[0]])]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.cli(command, *args)
        self.assertEqual((self.rep / '_work' / 'state.json').read_bytes(), original)

    def test_lock_rejects_other_writer_and_releases_after_exception(self):
        with self.assertRaisesRegex(ValueError, 'test'):
            with bt.output_lock(self.source):
                with self.assertRaises(RuntimeError):
                    with bt.output_lock(self.source):
                        pass
                raise ValueError('test')
        with bt.output_lock(self.source):
            pass

    def test_missing_input_returns_failure_without_api_call(self):
        self.fixture()
        (self.source / 'black' / f'{self.ids[0]}.png').unlink()
        self.assertEqual(self.cli('local', *self.ids), 1)

    def test_missing_tsv_id_does_not_submit(self):
        self.fixture()
        fake = FakeGemini()
        self.mock_api(fake)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli('regen', 'missing', '--tsv', str(self.tsv))
        self.assertEqual(fake.created, [])
        self.assertFalse((self.rep / '_work' / 'state.json').exists())

    def test_overlapping_directories_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            repair.main(['local', self.ids[0], '--source', str(self.source), '--repair', str(self.source / 'nested')])



