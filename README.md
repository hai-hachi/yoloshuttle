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


## Reproduce / continue training on the original dataset

The baseline `best.pt` came from Roboflow Universe project:

- workspace: `badyfriends`
- project: `badminton-shuttlecock-dv7zr`
- dataset version: `5`
- class: `Shuttlecock`

Version 5 contains 3,067 generated images and is the version referenced by the original model's `data.yaml`.

Create a free Roboflow API key, then set it in your shell.

PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_KEY"
```

Linux/macOS:

```bash
export ROBOFLOW_API_KEY="YOUR_KEY"
```

Download the exact dataset:

```bash
python download_original_dataset.py
```

Then continue fine-tuning the existing `best.pt` on the original data:

```bash
python train_finetune.py \
  --model best.pt \
  --data datasets/badyfriends_v5/data.yaml \
  --device 0 \
  --epochs 20 \
  --batch 16 \
  --lr0 0.0001 \
  --name original_v5_continue
```

For an RTX 2000 Ada laptop GPU, `batch=16` should be a reasonable first attempt for this nano model at 640 px. If CUDA runs out of memory, use `--batch 8`.

Because `best.pt` is already trained on this exact dataset, this run is continuation/fine-tuning rather than a new independent training run. The lower learning rate is intentional to avoid moving too far from the existing solution.

The output model is:

```text
runs/detect/original_v5_continue/weights/best.pt
```

Compare it against the original:

```bash
python test_shuttle.py best.pt
python test_shuttle.py runs/detect/original_v5_continue/weights/best.pt
```


## SCROBOT Gazebo synthetic dataset

The current SCROBOT debug branch can generate YOLO-format RGB images and labels
directly from Gazebo simulation truth.

Default generated location after cloning this repo to `~/Desktop/yoloshuttle`:

```text
dataset/gazebo_scrobot/
├── data.yaml
├── metadata.csv
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

The generator uses the same rendered shuttle STL, the simulated shuttle poses,
the robot/camera transform, and the live RGB `CameraInfo` to project YOLO
bounding boxes automatically.

The generated directory is ignored by Git.

## Ubuntu / CUDA smoke test

On the ThinkPad with the NVIDIA RTX 2000 Ada laptop GPU:

```bash
cd ~/Desktop/yoloshuttle
bash ubuntu_cuda_smoke.sh dataset/gazebo_scrobot/data.yaml
```

The script:

1. verifies `nvidia-smi`;
2. creates `.venv` if needed;
3. installs `requirements.txt`;
4. verifies that PyTorch can actually use CUDA;
5. prints the GPU and available VRAM;
6. if the generated dataset exists, performs a 1-epoch fine-tune from the
   repository `best.pt` using `device=0`, `imgsz=640`, and `batch=4`.

If the dataset does not exist yet, the driver/PyTorch CUDA check still runs and
the script exits after confirming CUDA readiness.

A successful training smoke writes:

```text
runs/detect/ubuntu_cuda_smoke/weights/best.pt
```

Do not replace the repository baseline `best.pt` from the smoke run. It is
only a compatibility test.

After CUDA and the dataset are verified, the first serious synthetic-only
fine-tune should remain conservative because the final deployment domain is the
real D435i camera, not Gazebo. A reasonable first experiment is:

```bash
python train_finetune.py \
  --model best.pt \
  --data dataset/gazebo_scrobot/data.yaml \
  --device 0 \
  --epochs 15 \
  --batch 8 \
  --imgsz 640 \
  --lr0 0.0001 \
  --name gazebo_scrobot_v1
```

Keep the original `best.pt` for comparison.
