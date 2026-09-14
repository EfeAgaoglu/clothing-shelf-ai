"""Read-only validation shared by shelf training and evaluation."""
from collections import Counter
from pathlib import Path
import hashlib
import math

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['top', 'outwear', 'sleeveless_top', 'shorts', 'trousers', 'skirt', 'dress']
DEFAULT_MODEL = ROOT / 'runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_config(path):
    path = project_path(path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Dataset YAML must contain a mapping.')
    names = data.get('names')
    if names != NAMES and names != dict(enumerate(NAMES)):
        raise ValueError(f'Class IDs must be exactly: {dict(enumerate(NAMES))}')
    root = Path(data.get('path', '.')).expanduser()
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    result = {'path': str(root), 'names': dict(enumerate(NAMES))}
    for split in ('train', 'val', 'test'):
        value = data.get(split)
        if value is None and split == 'test':
            continue
        if not isinstance(value, str):
            raise ValueError(f'{split}: a single images/<split> directory is required.')
        directory = (root / value).resolve()
        if directory.parent != root / 'images':
            raise ValueError(f'{split}: expected a direct child of {root / "images"}')
        result[split] = str(directory)
    return result


def validate_dataset(data, verify_images=True):
    errors, warnings, summary = [], [], {}
    seen_hashes = {}
    for split in ('train', 'val', 'test'):
        if split not in data:
            continue
        directory = Path(data[split])
        labels = Path(data['path']) / 'labels' / directory.name
        images = sorted(p for p in directory.glob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        counts, backgrounds, stems = Counter(), 0, set()
        if not images:
            errors.append(f'{split}: no images in {directory}')
        for p in images:
            if p.stem in stems:
                errors.append(f'{p}: duplicate image stem')
            stems.add(p.stem)
            try:
                if verify_images:
                    with Image.open(p) as im:
                        im.verify()
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                if digest in seen_hashes and seen_hashes[digest][0] != split:
                    errors.append(f'Split leakage: {p} == {seen_hashes[digest][1]}')
                seen_hashes[digest] = (split, str(p))
            except (OSError, ValueError) as exc:
                errors.append(f'{p}: unreadable image: {exc}')
            label = labels / f'{p.stem}.txt'
            if not label.is_file():
                errors.append(f'{label}: missing label (use an empty file only for a verified background)')
                continue
            try:
                lines = label.read_text(encoding='utf-8').splitlines()
            except (OSError, UnicodeError) as exc:
                errors.append(f'{label}: {exc}')
                continue
            if not any(line.strip() for line in lines):
                backgrounds += 1
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    tokens = line.split()
                    cls = int(tokens[0])
                    xy = list(map(float, tokens[1:]))
                    if not 0 <= cls < 7:
                        raise ValueError('class must be 0..6')
                    if len(xy) < 6 or len(xy) % 2:
                        raise ValueError('segmentation needs at least 3 x,y pairs; boxes are not masks')
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in xy):
                        raise ValueError('coordinates must be finite and normalized to [0,1]')
                    points = list(zip(xy[::2], xy[1::2]))
                    area2 = abs(sum(x*y2-x2*y for (x,y),(x2,y2) in zip(points, points[1:]+points[:1])))
                    if len(set(points)) < 3 or area2 <= 1e-12:
                        raise ValueError('degenerate polygon')
                    counts[cls] += 1
                except (ValueError, IndexError) as exc:
                    errors.append(f'{label}:{number}: {exc}')
        for label in labels.glob('*.txt'):
            if label.stem not in stems:
                errors.append(f'{label}: orphan label')
        if images and not sum(counts.values()):
            errors.append(f'{split}: no valid foreground polygons')
        missing = [name for i, name in enumerate(NAMES) if not counts[i]]
        if missing:
            warnings.append(f'{split}: classes without instances: {", ".join(missing)}')
        summary[split] = {'images': len(images), 'backgrounds': backgrounds,
                          'instances': {name: counts[i] for i, name in enumerate(NAMES)}}
    return {'ready': not errors, 'summary': summary, 'errors': errors, 'warnings': warnings}


def load_model(path):
    from ultralytics import YOLO
    path = project_path(path)
    if not path.is_file():
        raise ValueError(f'Checkpoint not found: {path}')
    model = YOLO(str(path))
    if model.task != 'segment' or model.names != dict(enumerate(NAMES)):
        raise ValueError('Checkpoint must be a segmentation model with the same seven class IDs.')
    return model


def choose_device(requested):
    import torch
    if requested != 'auto':
        return requested
    return '0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
