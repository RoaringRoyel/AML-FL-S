"""
check_setup.py
Run this FIRST from your project root:
  cd D:\Github\aml-with-fl\AML-with-FL
  python check_setup.py
"""

import sys, os
print("=" * 60)
print("AML PROJECT — ENVIRONMENT CHECK")
print("=" * 60)
print(f"\nWorking directory: {os.getcwd()}")

# ── Python ────────────────────────────────────────────────────────
print(f"\n✅ Python : {sys.version.split()[0]}")

# ── PyTorch + CUDA ────────────────────────────────────────────────
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu   = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda  = torch.version.cuda
        print(f"✅ CUDA   : YES ← you are using the GPU")
        print(f"   GPU    : {gpu}")
        print(f"   VRAM   : {vram:.1f} GB")
        print(f"   CUDA   : {cuda}")
        DEVICE = "cuda"
    else:
        print("⚠️  CUDA   : NOT detected — will use CPU (slow)")
        print("   Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128")
        DEVICE = "cpu"
except ImportError:
    print("❌ PyTorch not installed — run setup_env.bat first")
    sys.exit(1)

# ── PyTorch Geometric ─────────────────────────────────────────────
try:
    import torch_geometric
    print(f"✅ PyG    : {torch_geometric.__version__}")
except ImportError:
    print("❌ torch-geometric not installed")
    sys.exit(1)

# ── Flower ────────────────────────────────────────────────────────
try:
    import flwr
    print(f"✅ Flower : {flwr.__version__}")
except ImportError:
    print("❌ Flower not installed — pip install flwr[simulation]")
    sys.exit(1)

# ── MLflow ────────────────────────────────────────────────────────
try:
    import mlflow
    print(f"✅ MLflow : {mlflow.__version__}")
except ImportError:
    print("❌ MLflow not installed — pip install mlflow")
    sys.exit(1)

# ── Other deps ────────────────────────────────────────────────────
for dep in ["pandas", "numpy", "sklearn", "networkx", "streamlit", "fastapi", "plotly"]:
    try:
        mod = __import__(dep if dep != "sklearn" else "sklearn")
        print(f"✅ {dep:<12}: {getattr(mod,'__version__','ok')}")
    except ImportError:
        print(f"❌ {dep} not installed")

# ── Dataset files — matches your actual structure ─────────────────
print("\n── Dataset files ──")
files = {
    "data/HI-Small_Trans.csv":    ("USE FIRST",  "~455 MB"),
    "data/HI-Small_accounts.csv": ("USE FIRST",  "~34 MB"),
    "data/HI-Medium_Trans.csv":   ("later",      "~2.8 GB"),
    "data/LI-Small_Trans.csv":    ("evaluation", "~620 MB"),
    "data/LI-Small_accounts.csv": ("evaluation", "~45 MB"),
}
all_found = True
for path, (tag, size) in files.items():
    if os.path.exists(path):
        actual_mb = os.path.getsize(path) / 1e6
        print(f"✅ {path:<35} {actual_mb:>7.0f} MB  [{tag}]")
    else:
        print(f"❌ {path:<35} NOT FOUND  [{tag}]  expected {size}")
        if tag == "USE FIRST":
            all_found = False

if not all_found:
    print("\n⚠️  Copy HI-Small files to data/ before training.")

# ── SAGEConv GPU smoke test ────────────────────────────────────────
print("\n── GPU smoke test ──")
device = torch.device(DEVICE)
x  = torch.randn(500, 8).to(device)
ei = torch.randint(0, 500, (2, 1000)).to(device)
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
conv = SAGEConv(8, 16).to(device)
h = F.relu(conv(x, ei))
print(f"✅ SAGEConv on {DEVICE.upper()} — OK  output shape: {tuple(h.shape)}")

# ── MLflow write test ─────────────────────────────────────────────
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("_setup_test")
with mlflow.start_run(run_name="setup_check"):
    mlflow.log_param("check", True)
    mlflow.log_metric("dummy", 1.0)
print("✅ MLflow write — OK  (logs go to ./mlruns/)")

print("\n" + "=" * 60)
if DEVICE == "cuda":
    print(f"  🚀 You ARE using CUDA — RTX GPU detected")
    print(f"     {torch.cuda.get_device_name(0)}")
    print(f"     {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")
else:
    print("  ⚠️  Running on CPU — install CUDA PyTorch to use GPU")
print("=" * 60)

print("""
Run order:
  1.  python check_setup.py            ← you are here
  2.  python quick_start.py            ← 50k rows, 2-5 min smoke test
  3.  python run_centralized.py --nrows 500000 --epochs 80
  4.  python run_federated.py   --nrows 500000 --rounds 20
  5.  mlflow ui                        ← http://127.0.0.1:5000
  6.  streamlit run dashboard/app.py   ← http://localhost:8501
""")
