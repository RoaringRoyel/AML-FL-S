@echo off
REM ══════════════════════════════════════════════════════════════
REM  run_project.bat
REM  Exact commands for D:\Github\aml-with-fl\AML-with-FL\
REM
REM  Run one section at a time by reading the menu below.
REM  Or just double-click to see the menu.
REM ══════════════════════════════════════════════════════════════

cd /d D:\Github\aml-with-fl\AML-with-FL

echo.
echo  ============================================================
echo   AML Federated Project — Command Menu
echo   Project: D:\Github\aml-with-fl\AML-with-FL
echo  ============================================================
echo.
echo   [1] Setup venv (run ONCE)
echo   [2] Check CUDA and deps
echo   [3] Quick smoke test (2-5 min)
echo   [4] Centralized training  (1-2 hours)
echo   [5] Federated training    (1-2 hours)
echo   [6] Open MLflow dashboard
echo   [7] Open Streamlit dashboard
echo   [8] Open FastAPI
echo   [9] Docker (everything)
echo   [0] Exit
echo.
set /p choice="Enter number: "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto CHECK
if "%choice%"=="3" goto QUICK
if "%choice%"=="4" goto CENTRAL
if "%choice%"=="5" goto FEDERATED
if "%choice%"=="6" goto MLFLOW
if "%choice%"=="7" goto STREAMLIT
if "%choice%"=="8" goto API
if "%choice%"=="9" goto DOCKER
goto END

:SETUP
echo.
echo [SETUP] Creating .venv and installing all dependencies...
call setup_env.bat
goto END

:CHECK
echo.
echo [CHECK] Verifying CUDA and all dependencies...
call .venv\Scripts\activate
python check_setup.py
pause
goto END

:QUICK
echo.
echo [QUICK] Running 50k-row smoke test (2-5 minutes)...
call .venv\Scripts\activate
python quick_start.py ^
    --trans    data/HI-Medium_Trans.csv ^
    --accounts data/HI-Medium_accounts.csv ^
    --nrows    50000 ^
    --epochs   10
pause
goto END

:CENTRAL
echo.
echo [TRAINING] Centralized GraphSAGE baseline (500k rows, 80 epochs)
echo            Estimated time: 1-2 hours on RTX GPU
echo            MLflow: http://127.0.0.1:5000 (run option 6 in another terminal)
echo.
call .venv\Scripts\activate
python run_centralized.py ^
    --trans    data/HI-Medium_Trans.csv ^
    --accounts data/HI-Medium_accounts.csv ^
    --nrows    500000 ^
    --epochs   80 ^
    --lr       0.005 ^
    --save     outputs/centralized_model.pt
pause
goto END

:FEDERATED
echo.
echo [FEDERATED] Flower simulation — 4 banks, 20 rounds (500k rows)
echo             Estimated time: 1-2 hours on RTX GPU
echo.
call .venv\Scripts\activate
python run_federated.py ^
    --trans    data/HI-Medium_Trans.csv ^
    --accounts data/HI-Medium_accounts.csv ^
    --nrows    500000 ^
    --rounds   20 ^
    --clients  4 ^
    --epochs   5 ^
    --save     outputs/global_model.pt
pause
goto END

:MLFLOW
echo.
echo [MLFLOW] Starting MLflow UI...
echo          Open browser: http://127.0.0.1:5000
call .venv\Scripts\activate
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
goto END

:STREAMLIT
echo.
echo [DASHBOARD] Starting Streamlit...
echo             Open browser: http://localhost:8501
call .venv\Scripts\activate
streamlit run dashboard/app.py ^
    --server.port 8501 ^
    --server.address localhost
goto END

:API
echo.
echo [API] Starting FastAPI...
echo       Open browser: http://localhost:8000/docs
call .venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
goto END

:DOCKER
echo.
echo [DOCKER] Building and starting all containers...
echo          Make sure Docker Desktop is running.
docker compose up --build
pause
goto END

:END
echo.
pause
