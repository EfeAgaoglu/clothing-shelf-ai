"""Pack the prepared 10k subset as real files for Windows, following symlinks."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/fashionpedia_balanced_10k'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'transfer/fashionpedia_balanced_10k_windows.zip')
    args = parser.parse_args()
    files = []
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            directory = DATASET / kind / split
            entries = sorted(directory.iterdir())
            if not entries:
                raise ValueError(f'Empty directory: {directory}')
            for entry in entries:
                if not entry.is_file():
                    raise ValueError(f'Missing or invalid source: {entry}')
                files.append(entry)
    manifest = DATASET / 'subset_manifest.json'
    if manifest.is_file():
        files.append(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing transfer archive.
    with ZipFile(args.output, 'x', compression=ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            # ZipFile.write follows the source link and stores ordinary bytes.
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Archive CRC failed: {bad}')
    checksum = hashlib.sha256()
    with args.output.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    args.output.with_suffix('.zip.sha256').write_text(
        f'{checksum.hexdigest()}  {args.output.name}\n', encoding='utf-8')
    print(f'Exported {len(files)} files: {args.output}')
    print(f'Size: {args.output.stat().st_size / 1024**3:.2f} GiB')
    print(f'SHA256: {checksum.hexdigest()}')


if __name__ == '__main__':
    main()
