#!/usr/bin/env python3
"""Fine-tune YOLO on the annotated mikan dataset.

Example:
    python scripts/train.py --data data/mikan.yaml --model yolo11n.pt --epochs 100 --imgsz 640

The trained best.pt is copied into models/ so the Streamlit app can load it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import detector  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自前データでYOLOをファインチューニングする")
    parser.add_argument("--data", required=True, help="YOLO形式データセットの data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="ベースモデル（既定: yolo11n.pt）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="mikan", help="学習ラン名（runs/detect/<name>）")
    parser.add_argument("--device", default=None, help="未指定なら mps -> cuda -> cpu を自動選択")
    parser.add_argument("--out-name", default="best.pt", help="models/ にコピーするファイル名")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: data yaml not found: {data_path}", file=sys.stderr)
        return 1

    device = args.device or detector.select_device()
    model = detector.load_model(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=device,
    )

    best = Path(getattr(results, "save_dir", "runs/detect/" + args.name)) / "weights" / "best.pt"
    if best.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / args.out_name
        shutil.copy2(best, dest)
        print(f"copied {best} -> {dest}")
    else:
        print(f"warning: best.pt not found at {best}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
