#!/usr/bin/env python3
"""Fine-tune the current shuttlecock detector for SCROBOT low-angle images."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from plot_metrics import plot_metrics


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


def print_validation(metrics) -> None:
    print("\n" + "=" * 64)
    print("VALIDATION RESULT")
    print("=" * 64)
    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    print(f"mAP@50         : {metrics.box.map50:.4f}")
    print(f"mAP@75         : {metrics.box.map75:.4f}")
    print(f"mAP@50-95      : {metrics.box.map:.4f}")
    print(f"Fitness        : {metrics.box.fitness():.4f}")

    if getattr(metrics, "speed", None):
        print("\nSpeed (ms/image)")
        for key, value in metrics.speed.items():
            print(f"  {key:<12}: {value:.2f}")

    print("=" * 64)


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
        plots=True,
        verbose=True,
    )
    if args.device is not None:
        train_args["device"] = args.device

    results = model.train(**train_args)

    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    results_csv = save_dir / "results.csv"
    metrics_png = save_dir / "metrics.png"

    print("\nTraining complete.")
    print(f"Best weights: {best_path}")

    if results_csv.is_file():
        plot_metrics(results_csv, metrics_png)
    else:
        print(f"Warning: training results CSV not found: {results_csv}")

    print("\nRunning validation on best.pt ...")
    best_model = YOLO(str(best_path))

    val_args = dict(
        data=str(data_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        plots=True,
        verbose=True,
    )
    if args.device is not None:
        val_args["device"] = args.device

    metrics = best_model.val(**val_args)
    print_validation(metrics)

    print("\nGenerated plots:")
    print(f"  Training metrics : {metrics_png}")
    print(f"  Ultralytics plot : {save_dir / 'results.png'}")
    print("  Validation plots : PR_curve.png, F1_curve.png, P_curve.png, R_curve.png,")
    print("                     confusion_matrix.png (in the validation output directory)")

    print("\nTest live with:")
    print(f"  python test_shuttle.py {best_path}")


if __name__ == "__main__":
    main()
