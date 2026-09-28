#!/usr/bin/env python3
"""Validate any shuttlecock YOLO model and print the main metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a shuttlecock detector.")
    p.add_argument("model", help="Path to model weights, e.g. best.pt")
    p.add_argument("--data", default="datasets/badyfriends_v5/data.yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", choices=["val", "test"], default="val")
    args = p.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        verbose=True,
    )

    print("\n" + "=" * 64)
    print(f"VALIDATION RESULT: {model_path}")
    print(f"Split            : {args.split}")
    print("=" * 64)
    print(f"Precision        : {metrics.box.mp:.4f}")
    print(f"Recall           : {metrics.box.mr:.4f}")
    print(f"mAP@50           : {metrics.box.map50:.4f}")
    print(f"mAP@75           : {metrics.box.map75:.4f}")
    print(f"mAP@50-95        : {metrics.box.map:.4f}")
    print(f"Fitness          : {metrics.box.fitness():.4f}")

    if getattr(metrics, "speed", None):
        print("\nSpeed (ms/image)")
        for key, value in metrics.speed.items():
            print(f"  {key:<12}: {value:.2f}")

    print("=" * 64)


if __name__ == "__main__":
    main()
