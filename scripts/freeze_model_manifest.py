#!/usr/bin/env python3
"""Write a reproducibility manifest for a frozen SCROBOT YOLO model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics


REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        h.update(rel)
        h.update(b"\0")
        h.update(sha256_file(path).encode())
        h.update(b"\n")
    return h.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Freeze model and environment metadata.")
    p.add_argument("model", help="Path to the deployed .pt model")
    p.add_argument(
        "--dataset",
        default="dataset/gazebo_scrobot_simple",
        help="Dataset directory to fingerprint",
    )
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--output", default="artifacts/simulation_freeze_manifest.json")
    args = p.parse_args()

    model = Path(args.model).expanduser()
    if not model.is_absolute():
        model = REPO_ROOT / model
    dataset = Path(args.dataset).expanduser()
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output

    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "SCROBOT simulation freeze reproducibility record",
        "model": {
            "path": str(model.resolve()),
            "sha256": sha256_file(model),
            "size_bytes": model.stat().st_size,
        },
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256_tree": sha256_tree(dataset),
        },
        "inference": {
            "imgsz": args.imgsz,
            "conf": args.conf,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
