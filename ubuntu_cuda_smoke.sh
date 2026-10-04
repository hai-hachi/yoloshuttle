#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-dataset/gazebo_scrobot/data.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "============================================================"
echo "YOLOSHUTTLE Ubuntu/CUDA smoke test"
echo "repo     : $(pwd)"
echo "dataset  : ${DATA_PATH}"
echo "python   : ${PYTHON_BIN}"
echo "============================================================"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available. Install/repair the NVIDIA driver first."
  exit 2
fi

echo
echo "[1/5] NVIDIA driver/GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [ ! -d ".venv" ]; then
  echo
  echo "[2/5] Creating virtual environment"
  "${PYTHON_BIN}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo
echo "[3/5] Installing/updating Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "[4/5] Checking PyTorch CUDA"
python - <<'PY'
import sys
import torch

print("Python       :", sys.version.split()[0])
print("PyTorch      :", torch.__version__)
print("torch CUDA   :", torch.version.cuda)
print("CUDA usable  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available inside PyTorch. The NVIDIA driver may work, "
        "but this Python environment does not have a CUDA-enabled PyTorch build."
    )

print("GPU          :", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info(0)
print(f"VRAM free    : {free / 1024**3:.2f} GiB")
print(f"VRAM total   : {total / 1024**3:.2f} GiB")
PY

if [ ! -f "${DATA_PATH}" ]; then
  echo
  echo "CUDA/PyTorch check PASSED."
  echo "Dataset not found at: ${DATA_PATH}"
  echo "Generate it first with scrobot_debug/yolo_dataset_capture.launch.py,"
  echo "then rerun this script to include the one-epoch training smoke."
  exit 0
fi

echo
echo "Dataset split counts:"
for split in train val test; do
  img_dir="$(dirname "${DATA_PATH}")/images/${split}"
  label_dir="$(dirname "${DATA_PATH}")/labels/${split}"
  imgs=0
  labels=0
  [ -d "${img_dir}" ] && imgs="$(find "${img_dir}" -type f | wc -l)"
  [ -d "${label_dir}" ] && labels="$(find "${label_dir}" -type f | wc -l)"
  echo "  ${split}: images=${imgs} labels=${labels}"
done

echo
echo "[5/5] One-epoch CUDA fine-tuning smoke"
python train_finetune.py \
  --model best.pt \
  --data "${DATA_PATH}" \
  --device 0 \
  --epochs 1 \
  --batch 4 \
  --imgsz 640 \
  --lr0 0.0001 \
  --patience 2 \
  --workers 2 \
  --name ubuntu_cuda_smoke

echo
echo "============================================================"
echo "CUDA TRAINING SMOKE PASSED"
echo "weights: runs/detect/ubuntu_cuda_smoke/weights/best.pt"
echo "============================================================"
