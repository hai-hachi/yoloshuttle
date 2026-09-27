# YOLO Shuttlecock Detector

Small workspace for testing and fine-tuning shuttlecock detectors for the SCROBOT collector.

## Models currently in the repo

- `best.pt` - current baseline; this has performed best in the first low-angle concrete-floor test.
- `yolo11n_shuttle.pt` - comparison model.
- `yolo11s-ball.pt` - comparison model.

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\\Scripts\\activate   # Windows

pip install -r requirements.txt
```

## Test a model

Webcam:

```bash
python test_shuttle.py best.pt
```

Useful options:

```bash
python test_shuttle.py best.pt --camera 0 --conf 0.10 --imgsz 640
```

Press `q` to quit.

## Dataset layout

Put YOLO-format images and labels here:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image needs a YOLO label text file with the same stem:

```text
images/train/frame_001.jpg
labels/train/frame_001.txt
```

For this single-class dataset each label line is:

```text
0 x_center y_center width height
```

where the four box values are normalized to 0-1.

For SCROBOT, split by recording session/location rather than randomly splitting adjacent video frames. This avoids nearly identical frames appearing in train and validation.

## First fine-tuning test

Start from the existing `best.pt`.

Run a 3-epoch smoke test first:

```bash
python train_finetune.py --smoke-test
```

On an NVIDIA GPU:

```bash
python train_finetune.py --smoke-test --device 0
```

If that works, run the real fine-tune:

```bash
python train_finetune.py --epochs 50 --device 0
```

The trained model will be written to:

```text
runs/detect/scrobot_finetune/weights/best.pt
```

Compare it with the original:

```bash
python test_shuttle.py best.pt
python test_shuttle.py runs/detect/scrobot_finetune/weights/best.pt
```

## What to collect

Prioritize images from the actual RealSense mounting height and pitch. Include:

- 0.5-3 m distance
- cork toward camera, feathers toward camera, sideways, and random orientation
- concrete now, then the real green court when available
- white tape/court-line-like backgrounds
- shadows and bright lighting
- multiple shuttlecocks
- partial occlusion
- negative images with no shuttlecock
- examples the current model misses or falsely detects

The goal is not just higher confidence. Compare detection consistency and false positives at the distances needed by the robot.
