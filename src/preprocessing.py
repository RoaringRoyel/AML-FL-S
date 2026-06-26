"""
preprocessing.py
IBM AML Dataset — Load, clean, and split by bank.
Handles HI-Small, HI-Medium, HI-Large variants.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Expected column names (IBM AML dataset)
# ──────────────────────────────────────────────
TRANS_COLS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

ACCOUNT_COLS = ["Account", "Bank"]


# ──────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────

def load_transactions(path: str, nrows: int = None) -> pd.DataFrame:
    """Load *_Trans.csv.  Optionally limit rows for fast dev iteration."""
    log.info(f"Loading transactions from {path} (nrows={nrows})")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Drop rows with nulls in critical columns
    critical = ["Account", "Account.1", "Amount Received", "Is Laundering"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    log.info(f"Dropped {before - len(df)} null rows → {len(df):,} rows remain")

    # Cast types
    df["Amount Received"] = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0)
    df["Amount Paid"]     = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
    df["Is Laundering"]   = df["Is Laundering"].astype(int)

    # Create unique account IDs that include bank prefix (avoid collisions)
    df["src_id"] = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    df["dst_id"] = df["To Bank"].astype(str)   + "_" + df["Account.1"].astype(str)

    log.info(f"Laundering ratio: {df['Is Laundering'].mean():.4f}")
    log.info(f"Unique src accounts: {df['src_id'].nunique():,}")
    log.info(f"Unique dst accounts: {df['dst_id'].nunique():,}")
    return df


def load_accounts(path: str) -> pd.DataFrame:
    """Load *_accounts.csv."""
    log.info(f"Loading accounts from {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def build_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every unique account node compute 8 statistical features:
      1. out_degree       – number of outgoing transactions
      2. in_degree        – number of incoming transactions
      3. total_sent       – sum of Amount Paid (log1p scaled)
      4. total_received   – sum of Amount Received (log1p scaled)
      5. avg_sent         – mean Amount Paid (log1p)
      6. avg_received     – mean Amount Received (log1p)
      7. unique_out_peers – distinct counterparts sent to
      8. unique_in_peers  – distinct counterparts received from
    """
    # Outgoing stats
    out = df.groupby("src_id").agg(
        out_degree        = ("dst_id", "count"),
        total_sent        = ("Amount Paid", "sum"),
        avg_sent          = ("Amount Paid", "mean"),
        unique_out_peers  = ("dst_id", "nunique"),
    ).reset_index().rename(columns={"src_id": "node_id"})

    # Incoming stats
    inc = df.groupby("dst_id").agg(
        in_degree         = ("src_id", "count"),
        total_received    = ("Amount Received", "sum"),
        avg_received      = ("Amount Received", "mean"),
        unique_in_peers   = ("src_id", "nunique"),
    ).reset_index().rename(columns={"dst_id": "node_id"})

    # Merge
    feats = pd.merge(out, inc, on="node_id", how="outer").fillna(0)

    # Log-scale money features to reduce skew
    for col in ["total_sent", "total_received", "avg_sent", "avg_received"]:
        feats[col] = np.log1p(feats[col])

    log.info(f"Built features for {len(feats):,} nodes")
    return feats


def build_node_labels(df: pd.DataFrame, all_nodes: pd.Index) -> pd.Series:
    """
    Label = 1 if the account appears in ANY laundering transaction
    (either as sender or receiver).
    """
    launder_df = df[df["Is Laundering"] == 1]
    dirty_src = set(launder_df["src_id"].unique())
    dirty_dst = set(launder_df["dst_id"].unique())
    dirty = dirty_src | dirty_dst

    labels = pd.Series(
        [1 if n in dirty else 0 for n in all_nodes],
        index=all_nodes,
        dtype=np.int8,
    )

    pos = int(labels.sum())
    log.info(
        f"Labelled nodes → laundering: {pos:,} / {len(labels):,} "
        f"({pos/len(labels):.3f})"
    )
    return labels


# ──────────────────────────────────────────────
# Bank Splitting (for Federated Learning)
# ──────────────────────────────────────────────

def split_by_bank(df: pd.DataFrame, n_clients: int = 4) -> Dict[str, pd.DataFrame]:
    """
    Pick the top-N banks by transaction count and return one
    sub-DataFrame per bank (used as one Flower client each).
    """
    bank_counts = df["From Bank"].value_counts()
    top_banks   = bank_counts.index[:n_clients].tolist()
    log.info(f"Top {n_clients} banks: {top_banks}")

    splits: Dict[str, pd.DataFrame] = {}
    for bank in top_banks:
        sub = df[df["From Bank"] == bank].copy()
        splits[str(bank)] = sub
        log.info(f"  Bank {bank}: {len(sub):,} transactions")
    return splits


# ──────────────────────────────────────────────
# Convenience: one-call pipeline
# ──────────────────────────────────────────────

def full_pipeline(
    trans_path: str,
    accounts_path: str,
    nrows: int = None,
    n_clients: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns:
        df         – full transaction DataFrame
        node_feats – per-node feature DataFrame  (index = node_id)
        node_labels– per-node label Series       (index = node_id)
        bank_splits– dict  bank_name → transactions sub-DataFrame
    """
    df          = load_transactions(trans_path, nrows=nrows)
    node_feats  = build_node_features(df)
    node_feats  = node_feats.set_index("node_id")
    node_labels = build_node_labels(df, node_feats.index)
    bank_splits = split_by_bank(df, n_clients=n_clients)
    return df, node_feats, node_labels, bank_splits


if __name__ == "__main__":
    import sys
    trans    = sys.argv[1] if len(sys.argv) > 1 else "data/HI-Medium_Trans.csv"
    accounts = sys.argv[2] if len(sys.argv) > 2 else "data/HI-Medium_accounts.csv"
    df, feats, labels, splits = full_pipeline(trans, accounts)
    print("\n── Transaction sample ──")
    print(df.head(3))
    print("\n── Node features ──")
    print(feats.head(3))
    print("\n── Label distribution ──")
    print(labels.value_counts())
