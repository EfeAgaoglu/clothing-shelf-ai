"""Run the portable seven-class shelf checkpoint on one image or a directory."""
import argparse
from pathlib import Path

from shelf_utils import ROOT, choose_device, load_model


DEFAULT_MODEL = ROOT / "runs/shelf_7class/finetune_20260910_143218/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image, video or directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1024)
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.imgsz < 1:
        parser.error("--conf must be in (0,1] and --imgsz must be positive")
    source = Path(args.source).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        parser.error(f"Source not found: {source}")
    model = load_model(args.model)
    results = model.predict(source=str(source), device=choose_device(args.device),
                            conf=args.conf, imgsz=args.imgsz, iou=0.50,
                            max_det=300, save=True, project=str(ROOT / "runs/portable_predictions"))
    for result in results:
        print(f"{result.path}: {len(result.boxes)} detections")
        for box in result.boxes:
            print(f"  {model.names[int(box.cls[0])]} {float(box.conf[0]):.2%}")
        print(f"Saved to: {result.save_dir}")


if __name__ == "__main__":
    main()
