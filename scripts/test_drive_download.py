"""Offline download/extraction safety tests; no Drive files or training are used."""
import contextlib
import io
import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

from PIL import Image
import download_drive_assets as drive


class DownloadSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.assets = self.root / 'fixtures'
        self.assets.mkdir()
        self.archive = self.assets / 'dataset.zip'
        self.spec = {'kind': 'dataset', 'id': 'fake', 'path': 'transfer/dataset.zip',
                     'dataset': 'fixture', 'counts': {'train': 1, 'val': 1}}
        self.make_archive()

    def make_archive(self, extra=None):
        with ZipFile(self.archive, 'w') as archive:
            for split, color in [('train', 'red'), ('val', 'blue')]:
                buffer = io.BytesIO()
                Image.new('RGB', (16, 16), color).save(buffer, format='PNG')
                archive.writestr(f'data/fixture/images/{split}/one.png', buffer.getvalue())
                label = ''.join(f'{cls} .1 .1 .9 .1 .5 .9\n' for cls in range(7))
                archive.writestr(f'data/fixture/labels/{split}/one.txt', label)
            if extra:
                archive.writestr(extra, b'not allowed')
        self.spec['sha256'] = drive.sha256(self.archive)

    def fake_download(self, **kwargs):
        self.assertFalse(kwargs['use_cookies'])
        self.assertTrue(kwargs['verify'])
        shutil.copyfile(self.archive, kwargs['output'])
        return kwargs['output']

    def silent(self):
        return contextlib.redirect_stdout(io.StringIO())

    def test_download_and_rerun(self):
        with self.silent():
            path = drive.fetch_file(self.spec, self.root, self.fake_download)
            same = drive.fetch_file(self.spec, self.root, lambda **_: self.fail('Unexpected network call'))
        self.assertEqual(path, same)
        self.assertEqual(drive.sha256(path), self.spec['sha256'])
        self.assertEqual(list(path.parent.glob('.drive_download_*')), [])

    def test_existing_wrong_file_is_preserved(self):
        path = self.root / self.spec['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'user file')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            drive.fetch_file(self.spec, self.root, self.fake_download)
        self.assertEqual(path.read_bytes(), b'user file')

    def test_download_hash_mismatch_not_published(self):
        self.spec['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            drive.fetch_file(self.spec, self.root, self.fake_download)
        self.assertFalse((self.root / self.spec['path']).exists())

    def test_checksum_pin(self):
        path = self.assets / 'dataset.zip.sha256'
        spec = {'kind': 'checksum', 'expected_sha256': self.spec['sha256']}
        path.write_text(self.spec['sha256'] + '  dataset.zip\n', encoding='utf-8')
        self.assertEqual(drive.verify_file(path, spec), self.spec['sha256'])
        path.write_text('0' * 64 + '  dataset.zip\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            drive.verify_file(path, spec)

    def test_valid_extract_and_existing_data_retained(self):
        with self.silent():
            report = drive.restore_dataset(self.archive, self.spec, self.root)
            restored = self.root / 'data/fixture'
            original = (restored / 'labels/train/one.txt').read_bytes()
            drive.restore_dataset(self.archive, self.spec, self.root)
        self.assertTrue(report['ready'])
        self.assertEqual((restored / 'labels/train/one.txt').read_bytes(), original)
        self.assertEqual(list(restored.parent.glob('.dataset_restore_*')), [])

    def test_zip_slip_blocked_before_publication(self):
        self.make_archive('data/fixture/../../outside.txt')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            drive.restore_dataset(self.archive, self.spec, self.root)
        self.assertFalse((self.root / 'outside.txt').exists())
        self.assertFalse((self.root / 'data/fixture').exists())

    def test_unexpected_zip_file_blocked(self):
        self.make_archive('README.md')
        with self.assertRaisesRegex(ValueError, 'outside expected dataset'):
            drive.restore_dataset(self.archive, self.spec, self.root)
        self.assertFalse((self.root / 'README.md').exists())

    def test_zip_symlink_blocked(self):
        symlink = ZipInfo('data/fixture/link')
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.make_archive(symlink)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            drive.restore_dataset(self.archive, self.spec, self.root)

    def test_case_alias_blocked(self):
        self.make_archive('data/fixture/labels/train/ONE.txt')
        with self.assertRaisesRegex(ValueError, 'case-colliding'):
            drive.restore_dataset(self.archive, self.spec, self.root)

    def test_wrong_count_not_published(self):
        self.spec['counts']['train'] = 2
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            drive.restore_dataset(self.archive, self.spec, self.root)
        self.assertFalse((self.root / 'data/fixture').exists())

    def test_unsafe_destination_paths(self):
        for path in ('../outside', '/absolute', 'C:/outside', 'transfer\\outside', 'transfer/NUL.txt', 'transfer/foo.', 'transfer/foo:bar'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                drive.scoped_path(self.root, path)

    def test_plain_list_does_not_download(self):
        with patch('sys.argv', ['download_drive_assets.py']), patch.object(drive, 'fetch_file', side_effect=AssertionError('No network allowed')), self.silent():
            self.assertEqual(drive.main(), 0)

    def test_existing_dataset_skips_large_download(self):
        self.spec['checksum_id'] = 'fake_checksum'
        manifest = self.assets / 'manifest.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'assets': {'fashionpedia': self.spec}}), encoding='utf-8')
        with self.silent():
            drive.restore_dataset(self.archive, self.spec, self.root)

        def small_files_only(asset):
            self.assertEqual(asset['kind'], 'checksum', 'Existing dataset must not download another ZIP')
            return self.assets / 'unused_checksum_path'

        with patch('sys.argv', ['download_drive_assets.py', '--datasets', 'fashionpedia', '--extract', '--download']), patch.object(drive, 'ROOT', self.root), patch.object(drive, 'MANIFEST', manifest), patch.object(drive, 'fetch_file', side_effect=small_files_only), self.silent():
            self.assertEqual(drive.main(), 0)

    def test_checkpoints_require_pytorch_archive(self):
        path = self.assets / 'best.pt'
        with ZipFile(path, 'w') as archive:
            archive.writestr('checkpoint/data.pkl', b'not deserialized in this test')
        spec = {'kind': 'checkpoint', 'sha256': None}
        self.assertEqual(drive.verify_file(path, spec), drive.sha256(path))
        with ZipFile(path, 'w') as archive:
            archive.writestr('not-a-checkpoint.txt', b'x')
        with self.assertRaisesRegex(ValueError, 'Not a PyTorch'):
            drive.verify_file(path, spec)


if __name__ == '__main__':
    unittest.main()
