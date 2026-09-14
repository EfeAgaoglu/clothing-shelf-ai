"""Small synthetic regression tests; no training and no real dataset writes."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import yaml
from shelf_utils import NAMES, read_config, validate_dataset


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for split, color in [('train', 'red'), ('val', 'blue')]:
            (self.root / 'images' / split).mkdir(parents=True)
            (self.root / 'labels' / split).mkdir(parents=True)
            Image.new('RGB', (16, 16), color).save(self.root / 'images' / split / 'one.png')
            (self.root / 'labels' / split / 'one.txt').write_text('0 0.1 0.1 0.9 0.1 0.5 0.9\n')
        self.config = self.root / 'data.yaml'
        self.config.write_text(yaml.safe_dump(dict(path='.', train='images/train', val='images/val', names=NAMES)))

    def report(self):
        return validate_dataset(read_config(self.config))

    def test_valid_relative_paths(self):
        self.assertTrue(self.report()['ready'])

    def test_bad_polygons(self):
        for line in ['0 .5 .5 .2 .2', '7 0 0 1 0 0 1', '0 nan 0 1 0 0 1',
                     '0 0 0 2 0 0 1', '0 0 0 .5 .5 1 1', '0 0 0 1 0 0']:
            with self.subTest(line=line):
                (self.root / 'labels/train/one.txt').write_text(line)
                self.assertFalse(self.report()['ready'])

    def test_missing_label(self):
        (self.root / 'labels/train/one.txt').unlink()
        self.assertFalse(self.report()['ready'])

    def test_all_background_rejected(self):
        (self.root / 'labels/train/one.txt').write_text('')
        self.assertFalse(self.report()['ready'])

    def test_split_leakage(self):
        (self.root / 'images/val/one.png').write_bytes((self.root / 'images/train/one.png').read_bytes())
        self.assertTrue(any('Split leakage' in e for e in self.report()['errors']))

    def test_wrong_class_order(self):
        data = yaml.safe_load(self.config.read_text())
        data['names'] = NAMES[::-1]
        self.config.write_text(yaml.safe_dump(data))
        with self.assertRaises(ValueError):
            read_config(self.config)

    def test_duplicate_stem(self):
        Image.new('RGB', (16, 16)).save(self.root / 'images/train/one.jpg')
        self.assertFalse(self.report()['ready'])


if __name__ == '__main__':
    unittest.main()
