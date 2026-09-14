"""Fine-tune the seven-class checkpoint; training requires --execute."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import tempfile
import yaml
from shelf_utils import ROOT, DEFAULT_MODEL, read_config, validate_dataset, load_model, choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='shelf.yaml')
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Actually start training')
    mode.add_argument('--dry-run', action='store_true', help='Validate only (default)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--lr0', type=float, default=0.0001)
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1 or not 0 < args.lr0 <= 1:
        parser.error('epochs/imgsz/batch must be positive; lr0 must be in (0,1].')
    try:
        data = read_config(args.data)
        report = validate_dataset(data)
        model = load_model(args.model)
        options = dict(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                       device=choose_device(args.device), workers=0, optimizer='AdamW',
                       lr0=args.lr0, lrf=0.1, warmup_epochs=1.0, warmup_bias_lr=0.0,
                       patience=10, seed=42, resume=False, cache=False,
                       mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=5.0,
                       translate=0.05, scale=0.2, fliplr=0.5, flipud=0.0,
                       hsv_s=0.3, hsv_v=0.2, plots=True, exist_ok=False,
                       project=str(ROOT / 'runs/shelf_7class'),
                       name='finetune_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        from ultralytics.cfg import get_cfg
        get_cfg(overrides=options)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps({'dataset': report, 'training': options}, indent=2, ensure_ascii=False))
    if not report['ready']:
        print('Dataset is not ready. Training was not started.')
        return 2
    if not args.execute:
        print('Dry run passed. Training was not started; --execute is required.')
        return 0
    # Keep a unique resolved configuration for reproducibility and future validation.
    config_dir = ROOT / 'runs/shelf_7class/configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', prefix='dataset_',
                                     dir=config_dir, delete=False) as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        config_path = Path(handle.name)
    model.train(data=str(config_path), **options)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
