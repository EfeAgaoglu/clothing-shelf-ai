"""Validate images, YOLO polygons, class IDs and exact cross-split duplicates."""
import argparse
import json
from shelf_utils import read_config, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='shelf_7class.yaml')
    args = parser.parse_args()
    try:
        report = validate_dataset(read_config(args.data))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
