"""
train.py
Centralized and per-client training utilities.
Handles class imbalance using weighted cross-entropy.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from typing import Dict, Tuple

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Class weights for imbalanced AML data
# ──────────────────────────────────────────────

def compute_class_weights(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    In AML datasets laundering accounts are < 5% → this balances the loss.
    """
    counts  = y.bincount().float()
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)   # normalise
    log.info(f"Class weights: {weights.tolist()}")
    return weights.to(device)


# ──────────────────────────────────────────────
# Train / Eval helpers
# ──────────────────────────────────────────────

from torch_geometric.loader import NeighborLoader

def train_epoch(
    model,
    loader,
    optimizer,
    class_weights,
    device,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
        )

        # Only compute loss on seed nodes
        out = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]

        loss = F.cross_entropy(
            out,
            y,
            weight=class_weights,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_examples += y.size(0)

    return total_loss / total_examples

@torch.no_grad()
def evaluate(model, data_or_loader, device, mask=None):
    """
    Evaluate model on a Data object (full-graph) or a NeighborLoader (batched).

    Parameters
    ----------
    model          : GraphSAGE_AML
    data_or_loader : PyG Data  OR  NeighborLoader
    device         : torch.device or str
    mask           : optional bool tensor — used only when data_or_loader is a Data object
                     (pass data.val_mask / data.test_mask to restrict metrics to a split)
    """
    from torch_geometric.data import Data as PyGData

    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    if isinstance(data_or_loader, PyGData):
        # ── Full-graph inference (small/medium graphs, or per-client subgraphs) ──
        data = data_or_loader.to(device)
        out  = model(data.x, data.edge_index)
        prob = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        if mask is not None:
            mask = mask.to(device)
            y_true = data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            y_prob = prob[mask, 1].cpu().numpy()
        else:
            y_true = data.y.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_prob = prob[:, 1].cpu().numpy()

    else:
        # ── Batched inference via NeighborLoader (large graphs / run_centralized) ──
        all_y_true, all_y_pred, all_y_prob = [], [], []
        for batch in data_or_loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index)
            out   = out[:batch.batch_size]          # seed nodes only
            prob  = torch.softmax(out, dim=1)
            pred  = out.argmax(dim=1)
            all_y_true.append(batch.y[:batch.batch_size].cpu().numpy())
            all_y_pred.append(pred.cpu().numpy())
            all_y_prob.append(prob[:, 1].cpu().numpy())

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_prob = np.concatenate(all_y_prob)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = 0.0

    return metrics

def print_metrics(split: str, metrics: Dict[str, float]) -> None:
    log.info(
        f"{split:5s} → "
        f"Acc:{metrics['accuracy']:.4f}  "
        f"P:{metrics['precision']:.4f}  "
        f"R:{metrics['recall']:.4f}  "
        f"F1:{metrics['f1']:.4f}  "
        f"AUC:{metrics['auc']:.4f}"
    )


# ──────────────────────────────────────────────
# Full centralized training loop
# ──────────────────────────────────────────────

def train_centralized(
    model,
    data: Data,
    epochs: int    = 100,
    lr: float      = 0.005,
    patience: int  = 15,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[list, list]:
    """
    Train the model in a standard (non-federated) setting.
    Returns (train_losses, val_f1_scores).
    """
    device = torch.device(device_str)
    model  = model.to(device)
    data   = data.to(device)

    optimizer     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    class_weights = compute_class_weights(data.y, device)

    best_val_f1  = 0.0
    best_state   = None
    no_improve   = 0
    train_losses = []
    val_f1s      = []

    # ── Build loaders (train_centralized is used by quick_start.py) ─────────
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=True,
    )

    log.info(f"Starting centralized training for {epochs} epochs on {device}")

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, class_weights, device)
        train_losses.append(loss)

        if epoch % 5 == 0 or epoch == 1:
            val_m   = evaluate(model, data, device, mask=data.val_mask)
            train_m = evaluate(model, data, device, mask=data.train_mask)
            val_f1s.append(val_m["f1"])

            scheduler.step(val_m["f1"])

            log.info(f"Epoch {epoch:03d}/{epochs}  Loss:{loss:.4f}")
            print_metrics("TRAIN", train_m)
            print_metrics("VAL  ", val_m)

            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    test_m = evaluate(model, data, device, mask=data.test_mask)
    log.info("═" * 60)
    log.info("FINAL TEST RESULTS")
    print_metrics("TEST ", test_m)
    return train_losses, val_f1s


# ──────────────────────────────────────────────
# Per-client local training (used by Flower)
# ──────────────────────────────────────────────

from torch_geometric.loader import NeighborLoader

def local_train(
    model,
    data,
    epochs=5,
    lr=0.005,
    device_str="cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    class_weights = compute_class_weights(
        data.y,
        device,
    )

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=[15, 10],
        batch_size=4096,
        shuffle=False,
    )

    losses = []

    for _ in range(epochs):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            device,
        )
        losses.append(loss)

    val_m = evaluate(
        model,
        val_loader,
        device,
    )

    num_train = int(data.train_mask.sum())

    params = [
        p.cpu().detach().numpy()
        for p in model.parameters()
    ]

    return (
        params,
        num_train,
        {
            "loss": float(np.mean(losses)),
            **val_m,
        },
    )