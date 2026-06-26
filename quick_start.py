"""
quick_start.py
Runs a mini end-to-end test with 50,000 rows so you can
verify the FULL pipeline works before committing to hours of training.

Takes about 2-5 minutes on RTX 5060.

Usage:
    python quick_start.py
    python quick_start.py --trans data/HI-Medium_Trans.csv
"""

import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trans",    default="data/HI-Medium_Trans.csv")
    p.add_argument("--accounts", default="data/HI-Medium_accounts.csv")
    p.add_argument("--nrows",    type=int, default=50_000)
    p.add_argument("--epochs",   type=int, default=10)
    return p.parse_args()

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=" * 55)
    log.info(f"QUICK START TEST — device: {device.upper()}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Loading {args.nrows:,} rows from {args.trans}")
    log.info("=" * 55)

    # Step 1: Load data
    from src.preprocessing import full_pipeline
    df, feats, labels, splits = full_pipeline(
        trans_path    = args.trans,
        accounts_path = args.accounts,
        nrows         = args.nrows,
        n_clients     = 4,
    )
    log.info(f"Accounts (nodes): {len(feats):,}")
    log.info(f"Laundering nodes: {labels.sum():,} / {len(labels):,}  ({labels.mean():.2%})")

    # Step 2: Build graph
    from src.graph_builder import build_graph
    data, scaler, node_map = build_graph(df, feats, labels)
    log.info(f"Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    log.info(f"Feature dim: {data.x.shape[1]}")
    log.info(f"Class balance: {data.y.bincount().tolist()}")

    # Step 3: Build model
    from src.model import GraphSAGE_AML
    model = GraphSAGE_AML(in_channels=data.x.shape[1])
    log.info(f"Model parameters: {model.count_parameters():,}")

    # Step 4: Quick training (10 epochs)
    from src.train import train_centralized
    log.info(f"Training for {args.epochs} epochs on {device.upper()}...")
    losses, val_f1s = train_centralized(
        model      = model,
        data       = data,
        epochs     = args.epochs,
        device_str = device,
    )

    log.info("=" * 55)
    log.info("✅ QUICK START COMPLETE — pipeline works!")
    log.info(f"   Final val F1: {val_f1s[-1]:.4f}" if val_f1s else "")
    log.info("")
    log.info("Now run the full training:")
    log.info("  python run_centralized.py --nrows 500000 --epochs 80")
    log.info("  python run_federated.py   --nrows 500000 --rounds 20")
    log.info("=" * 55)

if __name__ == "__main__":
    main()
