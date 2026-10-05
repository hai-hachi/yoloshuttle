#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate a generated YOLO dataset and render bbox previews."
    )
    p.add_argument(
        "--dataset",
        default="dataset/gazebo_scrobot",
        help="Dataset root containing images/{train,val,test} and labels/{train,val,test}.",
    )
    p.add_argument("--samples", type=int, default=12, help="Number of preview images.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_images(path):
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def read_yolo_label(path):
    boxes = []
    errors = []

    if not path.exists():
        return boxes, [f"missing label: {path}"]

    text = path.read_text().strip()
    if not text:
        return boxes, errors  # Valid negative frame.

    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")
            continue

        try:
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric YOLO row")
            continue

        if cls != 0:
            errors.append(f"{path}:{line_no}: unexpected class id {cls}")

        for name, value in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{path}:{line_no}: {name}={value:.6f} outside [0,1]"
                )

        if w <= 0.0 or h <= 0.0:
            errors.append(f"{path}:{line_no}: non-positive width/height")

        boxes.append((cls, xc, yc, w, h))

    return boxes, errors


def draw_preview(image_path, label_path, out_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return False, f"could not read image: {image_path}"

    boxes, errors = read_yolo_label(label_path)
    h_img, w_img = image.shape[:2]

    for _, xc, yc, w, h in boxes:
        x0 = int(round((xc - w / 2.0) * w_img))
        y0 = int(round((yc - h / 2.0) * h_img))
        x1 = int(round((xc + w / 2.0) * w_img))
        y1 = int(round((yc + h / 2.0) * h_img))

        x0 = max(0, min(w_img - 1, x0))
        x1 = max(0, min(w_img - 1, x1))
        y0 = max(0, min(h_img - 1, y0))
        y1 = max(0, min(h_img - 1, y1))

        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            image,
            "Shuttlecock",
            (x0, max(16, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        return False, f"could not write preview: {out_path}"

    return True, errors


def main():
    args = parse_args()
    root = Path(args.dataset).expanduser().resolve()
    rng = random.Random(args.seed)

    if not root.exists():
        raise SystemExit(f"Dataset not found: {root}")

    print(f"Dataset: {root}")
    print()

    total_images = 0
    total_labels = 0
    total_boxes = 0
    total_negatives = 0
    all_pairs = []
    errors = []

    split_counts = {}

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        images = list_images(img_dir)
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)

        boxes_in_split = 0
        negatives = 0

        for image_path in images:
            label_path = lbl_dir / f"{image_path.stem}.txt"
            boxes, label_errors = read_yolo_label(label_path)
            errors.extend(label_errors)
            boxes_in_split += len(boxes)
            if len(boxes) == 0:
                negatives += 1
            all_pairs.append((split, image_path, label_path))

        split_counts[split] = len(images)
        total_images += len(images)
        total_labels += len(labels)
        total_boxes += boxes_in_split
        total_negatives += negatives

        print(
            f"{split:5s}: images={len(images):4d}  labels={len(labels):4d}  "
            f"boxes={boxes_in_split:5d}  negatives={negatives:4d}"
        )

        if missing_labels:
            errors.append(
                f"{split}: {len(missing_labels)} images missing label files"
            )
        if orphan_labels:
            errors.append(
                f"{split}: {len(orphan_labels)} labels without matching images"
            )

    print()
    if total_images:
        print(
            "Split ratio: "
            f"train={100.0*split_counts['train']/total_images:.1f}%  "
            f"val={100.0*split_counts['val']/total_images:.1f}%  "
            f"test={100.0*split_counts['test']/total_images:.1f}%"
        )
        print(
            f"Total: images={total_images} labels={total_labels} "
            f"boxes={total_boxes} negatives={total_negatives}"
        )
    else:
        errors.append("dataset contains no images")

    if split_counts.get("train", 0) == 0:
        errors.append("train split is empty")
    if split_counts.get("val", 0) == 0:
        errors.append("val split is empty")

    print()
    preview_root = root / "preview"
    preview_root.mkdir(parents=True, exist_ok=True)

    sample_count = min(max(0, args.samples), len(all_pairs))
    sampled = rng.sample(all_pairs, sample_count) if sample_count else []

    preview_errors = []
    for index, (split, image_path, label_path) in enumerate(sampled, 1):
        out_path = preview_root / f"{index:02d}_{split}_{image_path.name}"
        ok, detail = draw_preview(image_path, label_path, out_path)
        if not ok:
            preview_errors.append(str(detail))
        elif detail:
            preview_errors.extend(detail)

    errors.extend(preview_errors)

    print(f"Preview images: {preview_root}")
    print(f"Rendered samples: {sample_count}")

    print()
    if errors:
        print(f"CHECK RESULT: FAIL/WARN ({len(errors)} issue(s))")
        for item in errors[:30]:
            print(f"  - {item}")
        if len(errors) > 30:
            print(f"  ... plus {len(errors)-30} more")
        raise SystemExit(1)

    print("CHECK RESULT: PASS")
    print("Labels are structurally valid and train/val are non-empty.")
    print("Open the preview directory and visually confirm the green boxes cover shuttles.")


if __name__ == "__main__":
    main()
