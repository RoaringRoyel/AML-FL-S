"""
run_centralized.py  — Centralized baseline with MLflow tracking
"""

import argparse
import logging
import os
import json
import torch
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import NeighborLoader

from src.preprocessing  import full_pipeline
from src.graph_builder   import build_graph
from src.model           import GraphSAGE_AML
from src.train           import train_epoch, evaluate, compute_class_weights, print_metrics
from src.mlflow_tracker  import setup_mlflow, CentralizedTracker
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser("Centralized AML Training")
    p.add_argument("--trans",      default="data/HI-Medium_Trans.csv")
    p.add_argument("--accounts",   default="data/HI-Medium_accounts.csv")
    p.add_argument("--nrows",      type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=0.005)
    p.add_argument("--hidden",     type=int,   default=64)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--save",       default="outputs/centralized_model.pt")
    p.add_argument("--experiment", default="AML_Detection")
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


def plot_training(losses, val_f1s, out_path="outputs/centralized_training.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses);    ax1.set_title("Training Loss");      ax1.set_xlabel("Epoch"); ax1.grid(True)
    ax2.plot(val_f1s);   ax2.set_title("Val F1 (every 5ep)"); ax2.set_xlabel("×5 epochs"); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info(f"Plot → {out_path}")
    return out_path


def main():
    args   = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    os.makedirs("outputs", exist_ok=True)

    # MLflow experiment
    setup_mlflow(args.experiment)

    # Hyperparams to log
    params = {
        "model":        "GraphSAGE",
        "dataset":      os.path.basename(args.trans),
        "nrows":        args.nrows or "all",
        "epochs":       args.epochs,
        "lr":           args.lr,
        "hidden_dim":   args.hidden,
        "dropout":      args.dropout,
        "patience":     args.patience,
        "device":       device,
        "training_mode":"centralized",
    }

    log.info("=" * 60)
    log.info("CENTRALIZED BASELINE TRAINING")
    log.info(f"Device: {device.upper()}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info("=" * 60)

    # ── Data ───────────────────────────────────────────────────────
    df, feats, labels, _ = full_pipeline(args.trans, args.accounts, nrows=args.nrows)
    data, scaler, _      = build_graph(df, feats, labels)

    log.info(f"Nodes: {data.num_nodes:,}  Edges: {data.num_edges:,}")
    log.info(f"Class balance: {data.y.bincount().tolist()}")

    params["num_nodes"] = data.num_nodes
    params["num_edges"] = data.num_edges
    params["in_channels"] = int(data.x.shape[1])

    # ── Model ──────────────────────────────────────────────────────
    model = GraphSAGE_AML(
        in_channels  = data.x.shape[1],
        hidden_dim   = args.hidden,
        dropout      = args.dropout,
    )
    params["num_parameters"] = model.count_parameters()
    log.info(f"Model parameters: {model.count_parameters():,}")

    # ── Training loop with MLflow ──────────────────────────────────
    # ── Training loop with MLflow ──────────────────────────────────
    device_t = torch.device(device)
    model = model.to(device_t)

    # Keep the full graph on CPU
    data_d = data

    # Mini-batch neighbor sampling
    train_loader = NeighborLoader(
        data_d,
        input_nodes=data_d.train_mask,
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=True,
        num_workers=0,
    )

    val_loader = NeighborLoader(
        data_d,
        input_nodes=data_d.val_mask,
        num_neighbors=[15, 10],
        batch_size=4096,
        shuffle=False,
        num_workers=0,
    )

    test_loader = NeighborLoader(
        data_d,
        input_nodes=data_d.test_mask,
        num_neighbors=[15, 10],
        batch_size=4096,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=7,
        factor=0.5,
    )

    class_weights = compute_class_weights(
        data.y,
        device_t,
    )

    best_val_f1 = 0.0
    best_state = None
    no_improve = 0
    losses = []
    val_f1s = []

    with CentralizedTracker(
            params,
            run_name="centralized_graphsage"
    ) as tracker:

        for epoch in range(1, args.epochs + 1):

            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                class_weights,
                device_t,
            )

            losses.append(loss)

            if epoch % 5 == 0 or epoch == 1:

                train_m = evaluate(
                    model,
                    train_loader,
                    device_t,
                )

                val_m = evaluate(
                    model,
                    val_loader,
                    device_t,
                )

                val_f1s.append(val_m["f1"])

                tracker.log_epoch(
                    epoch,
                    loss,
                    train_m,
                    val_m,
                )

                scheduler.step(val_m["f1"])

                log.info(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"Loss:{loss:.4f}"
                )

                print_metrics("TRAIN", train_m)
                print_metrics("VAL  ", val_m)

                if val_m["f1"] > best_val_f1:
                    best_val_f1 = val_m["f1"]
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1

                    if no_improve >= args.patience:
                        log.info(
                            f"Early stopping at epoch {epoch}"
                        )
                        break

        if best_state:
            model.load_state_dict(best_state)

        test_m = evaluate(
            model,
            test_loader,
            device_t,
        )

        log.info("═" * 55)
        log.info("FINAL TEST RESULTS For Medium")
        print_metrics("TEST ", test_m)

        torch.save(
            {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "in_channels": data.x.shape[1],
            },
            args.save,
        )

        results = {
            "model": "GraphSAGE-centralized For Medium",
            **{
                k: round(v, 4)
                for k, v in test_m.items()
            },
        }

        with open(
                "outputs/centralized_results.json",
                "w",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
            )

        plot_path = plot_training(
            losses,
            val_f1s,
        )

        tracker.log_final(
            test_m,
            args.save,
            plot_path=plot_path,
        )

    log.info(f"\nMLflow UI: run  mlflow ui  → http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
