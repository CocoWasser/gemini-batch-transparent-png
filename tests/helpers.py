"""Synthetic image fixtures and an in-memory Gemini Batch API fake."""

import base64
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import batch_transparent as bt
import repair

PROJECT = Path(__file__).resolve().parents[1]


def pixels(background=255):
    alpha = np.zeros((128, 128), dtype=np.float32)
    alpha[32:96, 32:96] = 0.5
    alpha[40:88, 40:88] = 1
    color = np.zeros((128, 128, 3), dtype=np.float32)
    color[:] = (32, 144, 64)
    color[50:58, 50:58] = 255
    return np.rint(color * alpha[..., None] + background * (1-alpha[..., None])).astype('uint8')


def png_data(background):
    buf = io.BytesIO()
    Image.fromarray(pixels(background)).save(buf, format='PNG')
    return buf.getvalue()


class FakeGemini:
    def __init__(self, fail_after_create=False, errors=None):
        self.uploads = {}
        self.results = {}
        self.jobs = {}
        self.created = []
        self.closed = False
        self.fail_after_create = fail_after_create
        self.errors = {key: list(values) for key, values in (errors or {}).items()}
        self.files = SimpleNamespace(upload=self.upload)
        self.batches = SimpleNamespace(create=self.create, get=self.get, list=lambda: list(self.jobs.values()))

    def upload(self, file, config):
        name = f'files/input-{len(self.uploads)}'
        self.uploads[name] = [json.loads(line) for line in Path(file).read_text().splitlines()]
        return SimpleNamespace(name=name)

    def create(self, model, src, config):
        rows = self.uploads[src]
        is_black = any('inlineData' in part for part in rows[0]['request']['contents'][0]['parts'])
        bg = 0 if is_black else 255
        encoded = base64.b64encode(png_data(bg)).decode()
        result_rows = []
        stage = 'black' if is_black else 'white'
        for row in rows:
            errors = self.errors.get((stage, row['key']), [])
            if errors:
                code = errors.pop(0)
                result_rows.append({'key': row['key'], 'error': {
                    'code': code, 'message': 'Synthetic request failure'}})
            else:
                result_rows.append({
                    'key': row['key'],
                    'response': {'candidates': [{'content': {'parts': [
                        {'inlineData': {'data': encoded, 'mimeType': 'image/png'}}
                    ]}}]},
                })
        result = '\n'.join(json.dumps(row) for row in result_rows)
        name = f'batches/{len(self.created)}'
        file_name = f'files/result-{len(self.created)}'
        job = SimpleNamespace(name=name, display_name=config['display_name'],
                              state=SimpleNamespace(name='JOB_STATE_SUCCEEDED'),
                              dest=SimpleNamespace(file_name=file_name), error=None)
        self.results[file_name] = result
        self.jobs[name] = job
        self.created.append((model, is_black, rows))
        if self.fail_after_create:
            self.fail_after_create = False
            raise TimeoutError('simulated timeout after server created job')
        return job

    def get(self, name):
        return self.jobs[name]

    def download(self, file_name, result_file):
        Path(result_file).write_text(self.results[file_name], encoding='utf-8')

    def close(self):
        self.closed = True


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gemini-release-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.rep = self.root / 'repair'
        self.ids = ['asset-A_1', 'custom-02']
        self.tsv = self.root / 'input.tsv'
        self.tsv.write_text('ID\tName\tPrompt\n' + ''.join(
            f'{image_id}\tExample\tAn illustration on a transparent background\n'
            for image_id in self.ids
        ), encoding='utf-8')
        self.records = bt.load_tsv(self.tsv)
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network must remain disabled')).start()
        patch('httpx.Client.send', side_effect=AssertionError('Network must remain disabled')).start()
        patch('requests.sessions.Session.request', side_effect=AssertionError('Network must remain disabled')).start()
        patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-test-placeholder'}).start()
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def fixture(self):
        for stage, bg in [('white', 255), ('black', 0)]:
            (self.source / stage).mkdir(parents=True, exist_ok=True)
            for image_id in self.ids:
                (self.source / stage / f'{image_id}.png').write_bytes(png_data(bg))
        (self.source / 'transparent').mkdir(exist_ok=True)
        (self.source / '_work').mkdir(exist_ok=True)
        bt.initialize_state(self.records, self.source / '_work', 'source-model', '2K')

    def cli(self, command, *args):
        return repair.main([command, *args, '--source', str(self.source), '--repair', str(self.rep)])

    def mock_api(self, fake):
        patch.object(bt.genai, 'Client', return_value=fake).start()
        patch.object(bt, 'download_result_file', side_effect=fake.download).start()

