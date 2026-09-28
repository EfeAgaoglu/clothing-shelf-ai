"""List or explicitly download selected public Drive assets; never starts training."""
import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'drive_assets.json'


def scoped_path(root, relative):
    """Reject traversal and Windows path aliases before resolving symlinks."""
    parts = PurePosixPath(relative).parts
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{name}{i}' for name in ('COM', 'LPT') for i in range(1, 10)}
    if not parts or '\\' in relative or PurePosixPath(relative).is_absolute():
        raise ValueError(f'Unsafe relative path: {relative}')
    if any(p in ('.', '..') or ':' in p or '\x00' in p or p.endswith((' ', '.')) or p.split('.')[0].upper() in reserved for p in parts):
        raise ValueError(f'Unsafe relative path: {relative}')
    root = root.resolve()
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError(f'Path escapes destination: {relative}')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, asset):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Empty or missing download: {path}')
    if asset['kind'] == 'checksum':
        tokens = path.read_text(encoding='utf-8-sig').split()
        if not tokens or tokens[0].lower() != asset['expected_sha256']:
            raise ValueError(f'SHA-256 sidecar does not match the pinned hash: {path}')
        return tokens[0].lower()
    digest = sha256(path)
    if asset.get('sha256') and digest != asset['sha256']:
        raise ValueError(f'SHA-256 mismatch; existing files were not replaced: {path}')
    if asset['kind'] in ('dataset', 'checkpoint'):
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Archive CRC failure: {path}')
            if asset['kind'] == 'checkpoint' and not any(name.endswith('/data.pkl') for name in archive.namelist()):
                raise ValueError(f'Not a PyTorch checkpoint archive: {path}')
    elif asset['kind'] == 'image':
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    else:
        raise ValueError(f'Unknown asset type: {asset["kind"]}')
    return digest


def fetch_file(asset, root=ROOT, downloader=None):
    """Verify before publication; never overwrite an existing destination."""
    if not asset['path'].startswith('transfer/'):
        raise ValueError('Downloads must stay under transfer/.')
    target = scoped_path(root, asset['path'])
    if target.exists():
        digest = verify_file(target, asset)
        print(f'Existing verified file retained: {target}', flush=True)
        return target
    if downloader is None:
        try:
            import gdown
        except ImportError as exc:
            raise ValueError('Install requirements-download.txt with this Python interpreter first.') from exc
        downloader = gdown.download
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.drive_download_', dir=target.parent) as folder:
        temporary = Path(folder) / target.name
        result = downloader(id=asset['id'], output=str(temporary), quiet=False,
                            use_cookies=False, verify=True)
        if result is None:
            raise ValueError(f'Drive download failed; check permissions/quota for {asset["id"]}')
        digest = verify_file(temporary, asset)
        # Exclusive creation also protects files created concurrently.
        with temporary.open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    print(f'Downloaded: {target}\nSHA-256: {digest}', flush=True)
    if not asset.get('sha256') and asset['kind'] != 'checksum':
        print('Note: no publisher SHA-256 is pinned for this model/image; the printed hash is a local fingerprint, not proof of authenticity.', flush=True)
    return target


def checksum_asset(dataset):
    return {'kind': 'checksum', 'id': dataset['checksum_id'],
            'path': dataset['path'] + '.sha256', 'expected_sha256': dataset['sha256']}


def validate_restored(directory, expected_counts):
    from shelf_utils import NAMES, validate_dataset
    config = {'path': str(directory), 'names': dict(enumerate(NAMES)),
              **{split: str(directory / 'images' / split) for split in ('train', 'val')}}
    report = validate_dataset(config)
    if not report['ready'] or report['warnings']:
        raise ValueError(f'Dataset validation failed: {json.dumps(report, ensure_ascii=False)}')
    for split, expected in expected_counts.items():
        if report['summary'][split]['images'] != expected:
            raise ValueError(f'{split}: expected {expected} images, got {report["summary"][split]["images"]}')
    return report


def restore_dataset(archive_path, asset, root=ROOT):
    dataset = asset['dataset']
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', dataset):
        raise ValueError('Invalid dataset directory name.')
    target = scoped_path(root, f'data/{dataset}')
    if target.exists():
        report = validate_restored(target, asset['counts'])
        print(f'Existing dataset retained and validated: {target}', flush=True)
        return report
    verify_file(archive_path, asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        planned, seen, total_size = [], set(), 0
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            # Check even directory entries; allow only the two dataset ancestors.
            scoped_path(root, member.filename)
            if member.is_dir():
                if parts in (('data',), ('data', dataset)) or parts[:2] == ('data', dataset):
                    continue
                raise ValueError(f'Unexpected archive directory: {member.filename}')
            if parts[:2] != ('data', dataset) or len(parts) < 3:
                raise ValueError(f'File outside expected dataset: {member.filename}')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Symlink in archive: {member.filename}')
            key = '/'.join(parts).casefold()
            if key in seen:
                raise ValueError(f'Duplicate/case-colliding ZIP entry: {member.filename}')
            seen.add(key)
            total_size += member.file_size
            if total_size > 40 * 1024**3 or member.file_size > 512 * 1024**2:
                raise ValueError('Unexpectedly large uncompressed archive.')
            relative = PurePosixPath(*parts[2:]).as_posix()
            planned.append((member, relative))
        if not planned:
            raise ValueError('Archive contains no dataset files.')
        with tempfile.TemporaryDirectory(prefix='.dataset_restore_', dir=target.parent) as folder:
            staging = Path(folder) / dataset
            staging.mkdir()
            for member, relative in planned:
                destination = scoped_path(staging, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open('xb') as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
            report = validate_restored(staging, asset['counts'])
            if target.exists():
                raise FileExistsError(f'Dataset appeared concurrently; not replaced: {target}')
            staging.rename(target)
    print(f'Dataset extracted and validated: {target}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Select both Colab best.pt files')
    parser.add_argument('--images', action='store_true', help='Select the three original shelf images')
    parser.add_argument('--datasets', nargs='+', choices=('fashionpedia', 'deepfashion2'))
    parser.add_argument('--checksums-only', action='store_true', help='Select only the two small SHA-256 sidecars')
    parser.add_argument('--download', action='store_true', help='Download selected files (otherwise list only)')
    parser.add_argument('--extract', action='store_true', help='Extract and fully validate selected dataset ZIPs')
    args = parser.parse_args()
    if args.extract and not args.datasets:
        parser.error('--extract requires --datasets')
    if args.checksums_only and (args.models or args.images or args.datasets or args.extract):
        parser.error('--checksums-only cannot be combined with other selectors')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        parser.error('Unsupported manifest schema')
    selected = []
    for key, asset in manifest['assets'].items():
        if args.checksums_only and asset['kind'] == 'dataset':
            selected.append((key + '_checksum', checksum_asset(asset)))
        elif ((args.models and asset['kind'] == 'checkpoint') or
              (args.images and asset['kind'] == 'image') or
              (args.datasets and key in args.datasets)):
            if asset['kind'] == 'dataset':
                selected.append((key + '_checksum', checksum_asset(asset)))
            selected.append((key, asset))
    if not selected:
        if args.download:
            parser.error('Select --models, --images, --datasets or --checksums-only')
        selected = list(manifest['assets'].items())
    for key, asset in selected:
        print(f'{key}: {asset["path"]}\n  https://drive.google.com/file/d/{asset["id"]}/view', flush=True)
    if not args.download:
        print('List only. No download, extraction or training started; --download is required.')
        return 0
    if args.datasets or args.images:
        print('Dataset/image redistribution rights are not verified by this script. Follow the original dataset terms; public link access is not a license.', flush=True)
    try:
        for key, asset in selected:
            if args.extract and asset['kind'] == 'dataset':
                existing = scoped_path(ROOT, f'data/{asset["dataset"]}')
                if existing.exists():
                    report = validate_restored(existing, asset['counts'])
                    print(f'Existing dataset retained and validated; ZIP download skipped: {existing}', flush=True)
                    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
                    continue
            path = fetch_file(asset)
            if args.extract and asset['kind'] == 'dataset':
                report = restore_dataset(path, asset)
                print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\nExisting files were not intentionally replaced. Training was not started.\n')
    print('Selected files ready. Training was not started.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
