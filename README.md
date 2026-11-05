# YOLO Shuttlecock Detector

Training, evaluation, and deployment workspace for the SCROBOT shuttlecock detector.

## Folder map

```text
yoloshuttle/
├── README.md
├── requirements.txt
│
├── best.pt
├── yolo11n_shuttle.pt
├── yolo11s-ball.pt
│
├── scripts/
│   ├── train_finetune.py
│   ├── validate_model.py
│   ├── validate_generated_dataset.py
│   ├── test_shuttle.py
│   ├── plot_metrics.py
│   ├── download_original_dataset.py
│   └── ubuntu_cuda_smoke.sh
│
├── docs/
│   └── SCROBOT_YOLO_Training_Crash_Course.md
│
├── dataset/                 # local / generated datasets, ignored by Git
├── datasets/                # downloaded external datasets, ignored by Git
├── runs/                    # raw Ultralytics training runs, ignored by Git
└── artifacts/               # clean copies of trained models, ignored by Git
    ├── latest_model.txt
    └── models/
```

### What each folder is for

**`scripts/`**  
All executable helper programs live here. If you want to train, validate, test a camera, inspect a dataset, or download the original Roboflow dataset, start here.

**`docs/`**  
Human-readable notes. The YOLO crash course explains epochs, batch size, precision, recall, mAP, overfitting, target metrics, and the recommended SCROBOT workflow.

**`dataset/`**  
Locally generated datasets. The current Gazebo dataset is:

```text
dataset/gazebo_scrobot_simple/
```

This entire folder is ignored by Git.

**`datasets/`**  
Downloaded third-party datasets such as the original Roboflow dataset. Also ignored by Git.

**`runs/`**  
Raw Ultralytics experiment output. Every run gets its own folder:

```text
runs/detect/<run_name>/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── labels.jpg
└── ...
```

Do not treat this as the permanent model storage location.

**`artifacts/`**  
Clean model outputs that are easier to use later:

```text
artifacts/
├── latest_model.txt
└── models/
    ├── gazebo_simple_v1.pt
    └── gazebo_simple_v2.pt
```

After every completed training run, the script copies the best model here.

---

## Why the old path became duplicated

You previously saw:

```text
runs/detect/runs/detect/gazebo_simple_v2
```

The training script passed the relative string:

```text
project="runs/detect"
```

to Ultralytics while Ultralytics already had a runs directory context.

The reorganized script now converts the project path to an **absolute repository path** before calling Ultralytics.

New runs should therefore be exactly:

```text
runs/detect/gazebo_simple_v2
```

Existing old runs are not deleted automatically.

---

## Install

```bash
cd ~/Desktop/yoloshuttle

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1. Validate the Gazebo dataset

```bash
python scripts/validate_generated_dataset.py \
  --dataset dataset/gazebo_scrobot_simple \
  --samples 20
```

This checks:

- matching image and label files
- YOLO coordinate ranges
- train / val / test split counts
- duplicate images
- bbox preview images

---

## 2. CUDA smoke test

```bash
bash scripts/ubuntu_cuda_smoke.sh
```

This verifies the NVIDIA driver, PyTorch CUDA support, dataset availability, and one training epoch.

---

## 3. Train

### Conservative baseline run

```bash
python scripts/train_finetune.py \
  --model best.pt \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --epochs 15 \
  --batch 8 \
  --imgsz 640 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --patience 8 \
  --workers 4 \
  --name gazebo_simple_v1
```

### More aggressive small-object run

```bash
python scripts/train_finetune.py \
  --model artifacts/models/gazebo_simple_v1.pt \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --epochs 40 \
  --batch 8 \
  --imgsz 960 \
  --optimizer AdamW \
  --lr0 0.001 \
  --cos-lr \
  --patience 12 \
  --workers 4 \
  --name gazebo_simple_v2
```

The important output locations are then:

```text
runs/detect/gazebo_simple_v2/weights/best.pt
artifacts/models/gazebo_simple_v2.pt
artifacts/latest_model.txt
```

---

## 4. Validate a model

Newest trained model:

```bash
python scripts/validate_model.py latest \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --imgsz 960 \
  --batch 16 \
  --split test
```

By run name:

```bash
python scripts/validate_model.py gazebo_simple_v2 \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --split test
```

Baseline:

```bash
python scripts/validate_model.py best.pt \
  --data dataset/gazebo_scrobot_simple/data.yaml \
  --device 0 \
  --split test
```

---

## 5. Live camera test

Latest model:

```bash
python scripts/test_shuttle.py latest
```

Specific trained model:

```bash
python scripts/test_shuttle.py gazebo_simple_v2
```

Baseline:

```bash
python scripts/test_shuttle.py best.pt
```

Useful options:

```bash
python scripts/test_shuttle.py latest \
  --camera 0 \
  --conf 0.10 \
  --imgsz 960 \
  --width 1280 \
  --height 720
```

Press `q` to quit.

---

## 6. Original Roboflow dataset

```bash
export ROBOFLOW_API_KEY="YOUR_KEY"

python scripts/download_original_dataset.py
```

Default output:

```text
datasets/badyfriends_v5/
```

---

## Model-selection rule

Synthetic validation is useful, but the deployment target is the real D435i camera.

Always compare:

```text
baseline best.pt
        vs
new trained model
        ↓
same real green-court images
        ↓
compare:
- missed shuttlecocks
- false positives
- confidence
- detection range
- frame-to-frame stability
```

Do not replace the deployed baseline based only on Gazebo mAP.
