# 🔍 Federated Graph-Based AML Intelligence Platform

> **Anti-Money Laundering detection using GraphSAGE + Flower Federated Learning**  
> IBM AML Dataset · PyTorch Geometric · Flower AI · FastAPI · Streamlit · Docker · GitHub Actions CI/CD

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset Guide](#-dataset-guide)
- [Quick Start
- [Full Setup](#-full-setup)
- [Running the Project](#-running-the-project)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Report Guide](#-report-guide)

---

## 🎯 Project Overview

Traditional AML systems analyse transactions in isolation at individual banks. Real money laundering spans **multiple banks** through layered transfers — and banks legally cannot share raw customer data.

This platform solves both problems:

| Challenge | Solution |
|-----------|----------|
| Transactions form graphs, not tables | **GraphSAGE** — learns from account neighbours |
| Banks cannot share raw data | **Flower Federated Learning** — only model weights shared |
| Laundering accounts are < 5% of all accounts | **Weighted cross-entropy loss** — handles class imbalance |
| Need explainability | **Node feature importance** — explains risk scores |
| Production deployment | **Docker + GitHub Actions CI/CD** |

---

## 🏗️ Architecture

```
IBM AML Dataset (HI-Medium_Trans.csv)
          │
          ▼
   Preprocessing
   • Clean nulls
   • Build account IDs
   • Compute 8 node features
          │
          ▼
   Graph Construction (PyG)
   • Nodes = bank accounts
   • Edges = transactions
   • Labels = is_laundering (per node)
          │
   ┌──────┴──────┐
   │ Split by    │
   │ Bank (×4)   │
   └──────┬──────┘
          │
   ┌──────▼────────────────────────┐
   │     Flower Federated Learning  │
   │                                │
   │  BankA  BankB  BankC  BankD   │
   │  (GraphSAGE local training)    │
   │        ↓ weights only ↓        │
   │     FedAvg Aggregation         │
   │     (20 rounds)                │
   └──────────────┬─────────────────┘
                  │
          Global GraphSAGE Model
                  │
         ┌────────┴────────┐
         │                 │
    FastAPI              Streamlit
    (Inference)          (Dashboard)
         │
    Docker Compose
         │
    GitHub Actions CI/CD
```

---

## 📁 Dataset Guide

### Files and What They Mean

```
data/
├── HI-Small_Trans.csv      ← USE THIS FIRST (455 MB)
├── HI-Small_accounts.csv   ← USE THIS FIRST (34 MB)
├── HI-Medium_Trans.csv     ← Use for final experiments (2.8 GB)
├── HI-Medium_accounts.csv
├── LI-Small_Trans.csv      ← Realistic ratio (use for final eval)
├── LI-Small_accounts.csv
└── ...
```

### Naming Convention

```
HI  - Large  - Trans.csv
│       │        │
│       │        └── File type: Trans / accounts / Patterns
│       └─────────── Size: Small / Medium / Large
└─────────────────── Illicit ratio: HI (high ~10%) / LI (low ~0.1%)
```

### HI vs LI — Which to Use?

| File Type | When to Use |
|-----------|-------------|
| **HI-Small** | Development, debugging, quick training runs |
| **HI-Medium** | Final model training |
| **LI-Medium** | Final evaluation (realistic illicit ratio) |
| **\*-Large** | Research / paper experiments only |

### Column Reference (Trans.csv)

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | int | Transaction order |
| `From Bank` | str | Sending bank name |
| `Account` | str | Sender account ID |
| `To Bank` | str | Receiving bank name |
| `Account.1` | str | Receiver account ID |
| `Amount Received` | float | Amount at destination |
| `Receiving Currency` | str | Currency at destination |
| `Amount Paid` | float | Amount at source |
| `Payment Currency` | str | Currency at source |
| `Payment Format` | str | Wire / Cash / Cheque / etc. |
| **`Is Laundering`** | **0/1** | **Target label** |

### Node Features (8 features per account)

| Feature | Description |
|---------|-------------|
| `out_degree` | Number of outgoing transactions |
| `in_degree` | Number of incoming transactions |
| `total_sent` | Total amount sent (log1p) |
| `total_received` | Total amount received (log1p) |
| `avg_sent` | Average sent per transaction (log1p) |
| `avg_received` | Average received per transaction (log1p) |
| `unique_out_peers` | Distinct accounts sent to |
| `unique_in_peers` | Distinct accounts received from |

---

### Environment Setup (30 min)

```bash
# 1. Clone / unzip project
cd aml-federated

# 2. Copy CSV files
mkdir data
# Place HI-Small_Trans.csv and HI-Small_accounts.csv in data/

# 3. Create conda environment
conda create -n aml python=3.11
conda activate aml

# 4. Install PyTorch (RTX 5060 → CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Install PyG + dependencies
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu128.html

# 6. Install rest
pip install -r requirements.txt

# 7. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True  NVIDIA GeForce RTX 5060`

---

### Understand the Data

```bash
# Quick data exploration
python src/preprocessing.py data/HI-Small_Trans.csv data/HI-Small_accounts.csv
```

Expected output:
```
Loading transactions from data/HI-Small_Trans.csv (nrows=None)
Dropped 0 null rows → 5,078,941 rows remain
Laundering ratio: 0.0872
Unique src accounts: 14,321
Unique dst accounts: 15,204
```

---

### Centralized Baseline

```bash
# Train GraphSAGE without federation (baseline)
# --nrows 500000 for fast dev, remove for full dataset
python run_centralized.py \
    --trans    data/HI-Small_Trans.csv \
    --accounts data/HI-Small_accounts.csv \
    --nrows    500000 \
    --epochs   50

# Results saved to outputs/centralized_results.json
# Plot saved to outputs/centralized_training.png
```

---

### Federated Learning

```bash
# Run federated simulation (4 bank clients, 20 rounds)
python run_federated.py \
    --trans    data/HI-Small_Trans.csv \
    --accounts data/HI-Small_accounts.csv \
    --nrows    500000 \
    --rounds   20 \
    --clients  4 \
    --epochs   5

# Model saved to outputs/global_model.pt
# History saved to outputs/global_model_history.json
```

---

### Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

### API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs at http://localhost:8000/docs
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "BankA_8000ABC",
    "out_degree": 45,
    "in_degree": 3,
    "total_sent": 980000,
    "total_received": 5000,
    "avg_sent": 21777,
    "avg_received": 1666,
    "unique_out_peers": 40,
    "unique_in_peers": 2
  }'
```

---

### Docker

```bash
# Build and run all services
docker compose up --build

# Access:
#   Dashboard → http://localhost:8501
#   API       → http://localhost:8000
#   API docs  → http://localhost:8000/docs
```

---

## 🛠️ Full Setup

### Prerequisites

- Python 3.11+
- CUDA 12.8+ (for RTX 5060)
- Docker Desktop (for deployment)
- Git

### PyTorch Geometric Installation (critical — order matters)

```bash
# Step 1: PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Step 2: PyG core
pip install torch-geometric

# Step 3: PyG optional extensions (for SAGEConv)
pip install pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu128.html

# Step 4: Everything else
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "
import torch
import torch_geometric
import flwr
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('PyG:', torch_geometric.__version__)
print('Flower:', flwr.__version__)
"
```

---

## 🚀 Running the Project

### Option A — Simulation (recommended, single GPU)

```bash
# Full HI-Small dataset, 20 rounds
python run_federated.py \
    --trans    data/HI-Small_Trans.csv \
    --accounts data/HI-Small_accounts.csv \
    --rounds   20 \
    --clients  4 \
    --epochs   5

# Full HI-Medium dataset (better results, takes ~2-3 hours)
python run_federated.py \
    --trans    data/HI-Medium_Trans.csv \
    --accounts data/HI-Medium_accounts.csv \
    --rounds   20 \
    --clients  4
```

### Real Multi-Process (4 terminals)

**Terminal 1 — Server:**
```bash
python -m federated.server --rounds 20 --min-clients 4
```

**Terminals 2–5 — Clients:**
```bash
# Preprocess once, save to disk, then run clients
python run_client_docker.py  # set BANK_ID=0,1,2,3 in each terminal
```

### Option C — Docker (easiest for deployment)

```bash
docker compose up --build
```

---

## 🐳 Docker Deployment

### Build Images

```bash
docker compose build
```

### Run Services

```bash
# All services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f flower-server
docker compose logs -f bank-a

# Stop
docker compose down
```

### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | Streamlit UI |
| API | http://localhost:8000 | FastAPI endpoints |
| API Docs | http://localhost:8000/docs | Auto-generated Swagger UI |
| Flower Server | localhost:8080 | Internal FL server |

---

## 🔄 CI/CD Pipeline

### GitHub Actions — What Runs

```
git push origin main
        │
        ▼
  ┌─────────────┐
  │  Job 1:     │
  │  Lint+Test  │  ← flake8 + pytest (CPU, ~3 min)
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Job 2:     │
  │  Docker     │  ← Build all 3 images (~8 min)
  │  Build      │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Job 3:     │
  │  Push GHCR  │  ← Push to GitHub Container Registry
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Job 4:     │
  │  Deploy VPS │  ← SSH into server, docker compose up
  └─────────────┘
```

### Setup Secrets (GitHub → Settings → Secrets)

| Secret | Value |
|--------|-------|
| `DEPLOY_HOST` | Your VPS IP or domain |
| `DEPLOY_USER` | SSH username (e.g. `ubuntu`) |
| `DEPLOY_KEY` | SSH private key content |

### Run Tests Locally

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📂 Project Structure

```
aml-federated/
│
├── data/                          ← Put IBM AML CSV files here
│   ├── HI-Small_Trans.csv
│   ├── HI-Small_accounts.csv
│   └── ...
│
├── src/                           ← Core ML pipeline
│   ├── preprocessing.py           ← Data loading, feature engineering, bank split
│   ├── graph_builder.py           ← PyG Data objects from transactions
│   ├── model.py                   ← GraphSAGE architecture
│   └── train.py                   ← Training loop, evaluation metrics
│
├── federated/                     ← Flower FL components
│   ├── client.py                  ← Flower NumPyClient (one per bank)
│   └── server.py                  ← Flower server with FedAvg
│
├── api/                           ← FastAPI inference service
│   └── main.py
│
├── dashboard/                     ← Streamlit dashboard
│   └── app.py
│
├── docker/                        ← Dockerfiles
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── Dockerfile.dashboard
│
├── tests/                         ← Unit tests
│   └── test_pipeline.py
│
├── outputs/                       ← Generated after training
│   ├── centralized_model.pt
│   ├── centralized_results.json
│   ├── global_model.pt
│   └── global_model_history.json
│
├── .github/workflows/ci.yml       ← GitHub Actions CI/CD
├── docker-compose.yml
├── requirements.txt               ← Full (GPU)
├── requirements_docker.txt        ← Docker (CPU)
├── setup.py
├── run_centralized.py             ← Baseline training
├── run_federated.py               ← FL simulation
├── run_client_docker.py           ← Docker client entrypoint
└── README.md
```

---

## 📊 Results

> ⚠️ **These tables are empty until you run training.**
> After running, MLflow and `outputs/` will hold your real numbers.
> Fill in the table below with your actual results.

### Your Results (fill in after training)

```bash
# Run centralized baseline first:
python run_centralized.py --nrows 500000 --epochs 80

# Then federated:
python run_federated.py --nrows 500000 --rounds 20

# View results:
mlflow ui    →  http://127.0.0.1:5000
cat outputs/centralized_results.json


## 📝 Report Guide

### Key Points to Write in Your Report

**Why GNN over traditional ML?**
> Accounts form a graph. GraphSAGE aggregates information from transaction neighbours, capturing laundering patterns like cycles, fan-out, and smurfing that tabular models cannot detect.

**Why Federated Learning?**
> Banks are legally prohibited from sharing raw customer transaction data (GDPR, banking regulations). Federated Learning enables collaborative model training while ensuring no raw data leaves each institution's servers.

**Why Flower (flwr)?**
> Flower is a production-grade federated learning framework with built-in support for simulation, multi-process deployment, and custom aggregation strategies. It supports PyTorch natively.

**Class imbalance handling?**
> The IBM AML dataset has ~8-10% laundering transactions in HI variant and <1% in LI variant. We apply inverse-frequency class weighting to the cross-entropy loss, preventing the model from predicting all-normal.

**Explainability?**
> Node risk scores are computed from the model's output probability. Feature importance can be derived from edge weights in GraphSAGE aggregation. GNNExplainer (torch_geometric.explain) can be added for subgraph explanations.

---

## 🛠️ Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch by limiting rows
python run_federated.py --nrows 200000

# Or force CPU
python run_federated.py --cpu
```

### PyG Install Error
```bash
# Check torch version first
python -c "import torch; print(torch.__version__)"
# Then install matching PyG wheels
pip install pyg-lib -f https://data.pyg.org/whl/torch-{VERSION}+cu128.html
```

### Flower Clients Won't Connect
```bash
# Check server is running first
python -m federated.server --rounds 20

# Make sure all 4 clients start before timeout
# Default Flower wait = 2 minutes
```

### Docker: Port Already in Use
```bash
docker compose down
sudo lsof -i :8501 | awk 'NR!=1 {print $2}' | xargs kill
docker compose up -d
```
