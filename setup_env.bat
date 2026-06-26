@echo off
REM ══════════════════════════════════════════════════════════════
REM  setup_env.bat  —  AML Federated Project  (Windows)
REM  RTX 5060 · CUDA 12.8 · Python 3.11
REM
REM  Run this ONCE to create the virtual environment and install
REM  all dependencies.
REM
REM  Usage:
REM      Double-click setup_env.bat
REM      OR run in terminal:  .\setup_env.bat
REM ══════════════════════════════════════════════════════════════

echo.
echo  ============================================================
echo   AML Project — Environment Setup (Windows)
echo  ============================================================
echo.

REM ── Check Python version ──────────────────────────────────────
python --version 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.11 from python.org
    pause
    exit /b 1
)

REM ── Create virtual environment ────────────────────────────────
echo [1/7] Creating virtual environment in .venv ...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
)
echo       Done.

REM ── Activate ─────────────────────────────────────────────────
echo [2/7] Activating virtual environment ...
call .venv\Scripts\activate.bat
echo       Done. (.venv is now active)

REM ── Upgrade pip ───────────────────────────────────────────────
echo [3/7] Upgrading pip ...
python -m pip install --upgrade pip setuptools wheel
echo       Done.

REM ── Install PyTorch (CUDA 12.8 for RTX 5060) ─────────────────
echo [4/7] Installing PyTorch with CUDA 12.8 support ...
echo       (This is ~2 GB — may take a few minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch install failed.
    pause
    exit /b 1
)
echo       Done.

REM ── Install PyTorch Geometric ─────────────────────────────────
echo [5/7] Installing PyTorch Geometric ...
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster ^
    -f https://data.pyg.org/whl/torch-2.3.0+cu128.html
echo       Done.

REM ── Install project requirements ──────────────────────────────
echo [6/7] Installing project dependencies ...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] requirements.txt install failed.
    pause
    exit /b 1
)
echo       Done.

REM ── Install project as editable package ───────────────────────
echo [7/7] Installing project as editable package ...
pip install -e .
echo       Done.

REM ── Create data and outputs directories ───────────────────────
if not exist "data" mkdir data
if not exist "outputs" mkdir outputs
if not exist "mlruns" mkdir mlruns

REM ── Verify GPU ────────────────────────────────────────────────
echo.
echo  ── GPU Verification ──
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo.
echo  ============================================================
echo   Setup complete!
echo  ============================================================
echo.
echo  Next steps:
echo.
echo   1. Copy your CSV files into the data\ folder:
echo        data\HI-Medium_Trans.csv
echo        data\HI-Medium_accounts.csv
echo.
echo   2. Activate venv in any new terminal:
echo        .venv\Scripts\activate
echo.
echo   3. Run the check:
echo        python check_setup.py
echo.
echo   4. Quick test (50k rows, 10 epochs):
echo        python quick_start.py
echo.
echo   5. Full training:
echo        python run_centralized.py --nrows 500000
echo        python run_federated.py   --nrows 500000
echo.
echo   6. View MLflow dashboard:
echo        mlflow ui
echo        (open http://localhost:5000)
echo.
echo   7. View Streamlit dashboard:
echo        streamlit run dashboard\app.py
echo.
pause
