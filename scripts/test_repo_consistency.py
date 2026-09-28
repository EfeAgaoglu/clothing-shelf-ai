"""Offline checks for documentation links, code examples and retained CLI defaults."""
import argparse
import ast
import importlib
import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote
from unittest.mock import patch

import yaml
from shelf_utils import NAMES

ROOT = Path(__file__).resolve().parents[1]


def documents():
    return list(ROOT.glob('*.md')) + list((ROOT / 'data/external').rglob('*.md')) + list((ROOT / 'reports').rglob('*.md'))


def anchor(heading):
    return re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')


class CapturedParser(Exception):
    def __init__(self, parser):
        self.parser = parser


class RepoConsistency(unittest.TestCase):
    def test_local_markdown_links(self):
        for document in documents():
            for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8-sig')):
                if '://' in link:
                    continue
                filename, _, fragment = unquote(link).partition('#')
                target = (document.parent / filename).resolve() if filename else document
                with self.subTest(document=document.name, link=link):
                    self.assertTrue(target.exists(), f'Missing file: {target}')
                    if fragment and target.suffix == '.md':
                        headings = re.findall(r'^#{1,6} (.+)$', target.read_text(encoding='utf-8-sig'), re.M)
                        self.assertIn(fragment, [anchor(heading) for heading in headings])

    def test_python_document_examples(self):
        for document in documents():
            for code in re.findall(r'```python\n(.*?)```', document.read_text(encoding='utf-8-sig'), re.S):
                with self.subTest(document=document.name):
                    ast.parse(code, filename=str(document))

    def test_dataset_defaults_point_to_retained_template(self):
        def capture(parser, *args, **kwargs):
            raise CapturedParser(parser)

        for name in ('check_shelf_dataset', 'evaluate_shelf', 'preview_yolo_segmentation'):
            with self.subTest(script=name), patch.object(argparse.ArgumentParser, 'parse_args', capture):
                module = importlib.import_module(name)
                with self.assertRaises(CapturedParser) as caught:
                    module.main()
                default = caught.exception.parser.get_default('data')
                self.assertEqual(default, 'shelf.yaml')
                self.assertTrue((ROOT / default).is_file())
                if name == 'preview_yolo_segmentation':
                    self.assertEqual(caught.exception.parser.get_default('output'), 'runs/label_previews/preview.jpg')

    def test_root_dataset_class_order(self):
        for config in ROOT.glob('*.yaml'):
            with self.subTest(config=config.name):
                data = yaml.safe_load(config.read_text(encoding='utf-8-sig'))
                self.assertEqual(data['names'], dict(enumerate(NAMES)))

    def test_notebook_is_marked_as_history(self):
        notebook = json.loads((ROOT / 'notebooks/clothing_machine_learning.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['cells'][0]['cell_type'], 'markdown')
        banner = ''.join(notebook['cells'][0]['source'])
        self.assertIn('Arşiv', banner)
        self.assertIn('Tümünü çalıştır', banner)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell.get('outputs', []), [])


if __name__ == '__main__':
    unittest.main()
