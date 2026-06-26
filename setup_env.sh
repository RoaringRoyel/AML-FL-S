#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  setup_env.sh  —  AML Federated Project  (Linux / macOS)
#  RTX 5060 · CUDA 12.8 · Python 3.11
#
#  Usage:
#      chmod +x setup_env.sh
#      ./setup_env.sh
# ══════════════════════════════════════════════════════════════════

set -e   # exit immediately on error

echo ""
echo "============================================================"
echo " AML Project — Environment Setup (Linux/macOS)"
echo "============================================================"
echo ""

# ── Check Python ─────────────────────────────────────────────────
if ! command -v python3.11 &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3.11 not found. Install via:"
    echo "  sudo apt install python3.11 python3.11-venv    # Ubuntu"
    echo "  brew install python@3.11                        # macOS"
    exit 1
fi

PYTHON=$(command -v python3.11 || command -v python3)
echo "[INFO] Using Python: $($PYTHON --version)"

# ── Create virtual environment ────────────────────────────────────
echo ""
echo "[1/7] Creating virtual environment in .venv ..."
$PYTHON -m venv .venv
echo "      Done."

# ── Activate ──────────────────────────────────────────────────────
echo "[2/7] Activating .venv ..."
source .venv/bin/activate
echo "      Done."

# ── Upgrade pip ───────────────────────────────────────────────────
echo "[3/7] Upgrading pip ..."
pip install --upgrade pip setuptools wheel
echo "      Done."

# ── PyTorch (CUDA 12.8) ───────────────────────────────────────────
echo "[4/7] Installing PyTorch (CUDA 12.8) ..."
echo "      (~2 GB download, may take a few minutes)"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
echo "      Done."

# ── PyTorch Geometric ─────────────────────────────────────────────
echo "[5/7] Installing PyTorch Geometric ..."
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.3.0+cu128.html
echo "      Done."

# ── Project requirements ──────────────────────────────────────────
echo "[6/7] Installing project dependencies ..."
pip install -r requirements.txt
echo "      Done."

# ── Editable install ──────────────────────────────────────────────
echo "[7/7] Installing project as editable package ..."
pip install -e .
echo "      Done."

# ── Directories ───────────────────────────────────────────────────
mkdir -p data outputs mlruns

# ── GPU check ─────────────────────────────────────────────────────
echo ""
echo "── GPU Verification ──"
python -c "
import torch
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
    print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
else:
    print('GPU            : NONE — will run on CPU')
"

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " Activate venv in future terminals:"
echo "   source .venv/bin/activate"
echo ""
echo " Next steps:"
echo "   python check_setup.py"
echo "   python quick_start.py"
echo "   mlflow ui   →  http://localhost:5000"
echo ""
