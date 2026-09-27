#!/usr/bin/env python3
"""Fine-tune the current shuttlecock detector for SCROBOT low-angle images."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune shuttlecock YOLO weights.")
    p.add_argument("--model", default="best.pt", help="Starting weights (default: best.pt)")
    p.add_argument("--data", default="dataset/data.yaml", help="YOLO dataset YAML")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps; default: auto")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="scrobot_finetune")
    p.add_argument("--smoke-test", action="store_true", help="Train only 3 epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    epochs = 3 if args.smoke_test else args.epochs

    print("=" * 64)
    print("SCROBOT shuttle detector fine-tuning")
    print(f"model      : {model_path}")
    print(f"data       : {data_path}")
    print(f"epochs     : {epochs}")
    print(f"imgsz      : {args.imgsz}")
    print(f"batch      : {args.batch}")
    print(f"lr0        : {args.lr0}")
    print(f"device     : {args.device or 'auto'}")
    print("=" * 64)

    model = YOLO(str(model_path))

    train_args = dict(
        data=str(data_path),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers,
        project="runs/detect",
        name=args.name,
        exist_ok=True,
        seed=42,
        close_mosaic=10,
        verbose=True,
    )
    if args.device is not None:
        train_args["device"] = args.device

    results = model.train(**train_args)

    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"

    print("\nTraining complete.")
    print(f"Best weights: {best_path}")
    print("Test with:")
    print(f"  python test_shuttle.py {best_path}")


if __name__ == "__main__":
    main()
