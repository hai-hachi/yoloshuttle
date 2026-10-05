# YOLO Shuttlecock Detector

Training and evaluation workspace for the SCROBOT shuttlecock detector.

The repository keeps only source code and baseline weights. Generated datasets,
Ultralytics runs, and trained model artifacts are local-only and ignored by Git.

## Repository layout

```text
yoloshuttle/
├── best.pt                         # current baseline model
├── yolo11n_shuttle.pt              # comparison weights
├── yolo11s-ball.pt                 # comparison weights
├── train_finetune.py               # training / fine-tuning
├── validate_model.py               # val/test metrics
├── validate_generated_dataset.py   # dataset structure + bbox previews
├── test_shuttle.py                 # live camera test
├── plot_metrics.py                 # training metric plots
├── download_original_dataset.py    # original Roboflow dataset helper
├── ubuntu_cuda_smoke.sh            # CUDA + one-epoch smoke test
├── requirements.txt
│
├── dataset/                        # generated/local datasets, gitignored
├── datasets/                       # downloaded external datasets, gitignored
├── runs/                           # raw Ultralytics runs, gitignored
└── artifacts/                      # stable trained-model copies, gitignored
    ├── latest_model.txt
    └── models/<run_name>.pt
```

## Install

```bash
cd ~/Desktop/yoloshuttle

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Current SCROBOT Gazebo dataset

The current generator is the simple native-bbox pipeline in the SCROBOT
`yolo-simple-capture` branch.

Default dataset:

```text
dataset/gazebo_scrobot_simple/
├── data.yaml
├── metadata.csv
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

All dataset contents are ignored by Git.

Validate the generated dataset before training:

```bash
python validate_generated_dataset.py \
  --dataset dataset/gazebo_scrobot_simple \
  --samples 20
```

The validator checks image/label pairs, YOLO ranges, split counts, exact image
duplicates, and regenerates bbox previews under the dataset's `preview/`
directory.

## CUDA smoke test

On the Ubuntu NVIDIA machine:

```bash
bash ubuntu_cuda_smoke.sh
```

The script checks the NVIDIA driver, PyTorch CUDA support, dataset split counts,
and runs one training epoch.

Smoke-test outputs:

```text
runs/detect/ubuntu_cuda_smoke/weights/best.pt
artifacts/models/ubuntu_cuda_smoke.pt
```

## Train on the Gazebo dataset

Use the current real-world baseline `best.pt` as the starting weights.

A conservative first synthetic fine-tune:

```bash
python train_finetune.py \
  --model best.pt \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --epochs 15 \
  --batch 8 \
  --imgsz 640 \
  --lr0 0.0001 \
  --patience 8 \
  --workers 4 \
  --name gazebo_simple_v1
```

Each completed training run keeps the original Ultralytics directory and also
creates a predictable stable copy:

```text
runs/detect/gazebo_simple_v1/weights/best.pt
artifacts/models/gazebo_simple_v1.pt
artifacts/latest_model.txt
```

`artifacts/latest_model.txt` points to the newest completed training model.

## Validate a trained model

You no longer need to remember the full `runs/detect/...` path.

Validate the newest trained model:

```bash
python validate_model.py latest \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --imgsz 640 \
  --batch 16 \
  --split test
```

Or validate by run name:

```bash
python validate_model.py gazebo_simple_v1 \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --split test
```

Or provide a normal weight path:

```bash
python validate_model.py best.pt \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --split test
```

If the requested model does not exist, the validator prints the trained models
it can find under `artifacts/models/` and `runs/detect/`.

For a useful comparison, evaluate both the baseline and fine-tuned model on the
same test split.

## Live camera test

Baseline:

```bash
python test_shuttle.py best.pt
```

Latest trained model:

```bash
python test_shuttle.py latest
```

Named run:

```bash
python test_shuttle.py gazebo_simple_v1
```

Useful options:

```bash
python test_shuttle.py latest \
  --camera 0 \
  --conf 0.10 \
  --imgsz 640 \
  --width 1280 \
  --height 720
```

Press `q` to quit.

## Original Roboflow dataset

The repository baseline originated from Roboflow Universe:

- workspace: `badyfriends`
- project: `badminton-shuttlecock-dv7zr`
- version: `5`
- class: `Shuttlecock`

Download it with:

```bash
export ROBOFLOW_API_KEY="YOUR_KEY"
python download_original_dataset.py
```

It is written under `datasets/`, which is ignored by Git.

Example continuation run:

```bash
python train_finetune.py \
  --model best.pt \
  --data datasets/badyfriends_v5/data.yaml \
  --device 0 \
  --epochs 20 \
  --batch 16 \
  --imgsz 640 \
  --lr0 0.0001 \
  --name original_v5_continue
```

## Evaluation rule

Synthetic metrics are useful for checking whether the model learned the Gazebo
domain, but the final deployment target is the real D435i camera. Always compare
the baseline and fine-tuned models on real green-court images before replacing
`best.pt`.
