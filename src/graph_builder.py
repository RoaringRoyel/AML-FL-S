"""
graph_builder.py
Convert an IBM-AML transaction DataFrame into a PyTorch Geometric Data object.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from typing import Optional

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "out_degree",
    "in_degree",
    "total_sent",
    "total_received",
    "avg_sent",
    "avg_received",
    "unique_out_peers",
    "unique_in_peers",
]


def build_graph(
    df: pd.DataFrame,
    node_feats: pd.DataFrame,
    node_labels: pd.Series,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> tuple[Data, StandardScaler, dict]:
    """
    Build a PyG Data object from transactions.

    Parameters
    ----------
    df          : full transaction DataFrame  (must have src_id, dst_id columns)
    node_feats  : DataFrame indexed by node_id with FEATURE_COLS
    node_labels : Series indexed by node_id  (0 / 1)
    scaler      : reuse a fitted scaler (for client graphs that share global scale)
    fit_scaler  : if True, fit scaler on this data

    Returns
    -------
    data        : PyG Data(x, edge_index, y, train_mask, val_mask, test_mask)
    scaler      : the (fitted) StandardScaler
    node_map    : dict  node_id → integer index
    """
    # ── 1. Build node index ─────────────────────────────────────────
    all_nodes = node_feats.index.tolist()
    node_map  = {n: i for i, n in enumerate(all_nodes)}
    N         = len(all_nodes)
    log.info(f"Graph: {N:,} nodes")

    # ── 2. Node feature matrix ──────────────────────────────────────
    X = node_feats.reindex(all_nodes)[FEATURE_COLS].fillna(0).values.astype(np.float32)
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
    if fit_scaler and scaler is not None:
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)

    # ── 3. Edge index ────────────────────────────────────────────────
    # Keep only edges whose both endpoints exist in node_map
    valid_mask = df["src_id"].isin(node_map) & df["dst_id"].isin(node_map)
    df_valid   = df[valid_mask]

    src_idx = df_valid["src_id"].map(node_map).values
    dst_idx = df_valid["dst_id"].map(node_map).values

    edge_index = torch.tensor(
        np.stack([src_idx, dst_idx], axis=0), dtype=torch.long
    )
    log.info(f"Graph: {edge_index.shape[1]:,} edges")

    # ── 4. Edge features (optional) ──────────────────────────────────
    # Amount Paid (log1p) and Is Laundering flag per edge
    amounts  = np.log1p(df_valid["Amount Paid"].values).astype(np.float32)
    edge_lbl = df_valid["Is Laundering"].values.astype(np.float32)
    edge_attr = torch.tensor(
        np.stack([amounts, edge_lbl], axis=1), dtype=torch.float
    )

    # ── 5. Node labels ───────────────────────────────────────────────
    y_list = node_labels.reindex(all_nodes).fillna(0).astype(int).values
    y = torch.tensor(y_list, dtype=torch.long)

    # ── 6. Train / Val / Test masks ──────────────────────────────────
    idx    = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)

    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    train_mask[idx[:n_train]]            = True
    val_mask[idx[n_train:n_train+n_val]] = True
    test_mask[idx[n_train+n_val:]]       = True

    log.info(
        f"Masks → train:{train_mask.sum()}, "
        f"val:{val_mask.sum()}, test:{test_mask.sum()}"
    )

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = y,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )
    return data, scaler, node_map


def build_bank_graph(
    bank_df: pd.DataFrame,
    global_node_feats: pd.DataFrame,
    global_node_labels: pd.Series,
    scaler: StandardScaler,
) -> Data:
    """
    Build a subgraph for one Flower client (one bank).
    Uses the globally computed node features so all clients share
    the same feature space — critical for FedAvg weight averaging.
    """
    # Only keep accounts that appear in this bank's transactions
    local_accounts = set(bank_df["src_id"]) | set(bank_df["dst_id"])
    local_feats  = global_node_feats[global_node_feats.index.isin(local_accounts)]
    local_labels = global_node_labels[global_node_labels.index.isin(local_accounts)]

    data, _, _ = build_graph(
        df          = bank_df,
        node_feats  = local_feats,
        node_labels = local_labels,
        scaler      = scaler,
        fit_scaler  = False,   # use the pre-fitted global scaler
    )
    return data


if __name__ == "__main__":
    from src.preprocessing import full_pipeline

    df, feats, labels, splits = full_pipeline(
        "data/HI-Medium_Trans.csv",
        "data/HI-Medium_accounts.csv",
        nrows=200_000,
    )
    data, scaler, node_map = build_graph(df, feats, labels)
    print(data)
    print("Class balance:", data.y.bincount())
