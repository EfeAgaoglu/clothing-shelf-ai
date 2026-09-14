"""Evaluate a checkpoint on annotated shelf validation/test images."""
import argparse
import tempfile
from pathlib import Path
import yaml
from shelf_utils import ROOT, DEFAULT_MODEL, read_config, validate_dataset, load_model, choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='shelf_7class.yaml')
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--imgsz', type=int, default=512)
    args = parser.parse_args()
    try:
        data = read_config(args.data)
        if args.split not in data:
            raise ValueError(f'{args.split} is not configured in the dataset YAML.')
        report = validate_dataset(data)
        if not report['ready']:
            raise ValueError('\n'.join(report['errors']))
        model = load_model(args.model)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    with tempfile.TemporaryDirectory(prefix='shelf_eval_') as directory:
        config = Path(directory) / 'data.yaml'
        config.write_text(yaml.safe_dump(data), encoding='utf-8')
        model.val(data=str(config), split=args.split, device=choose_device(args.device),
                  imgsz=args.imgsz, batch=1, workers=0,
                  project=str(ROOT / 'runs/shelf_7class'), name='evaluation', exist_ok=False)


if __name__ == '__main__':
    main()
