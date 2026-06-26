"""
tests/test_pipeline.py
Unit tests for preprocessing, graph building, and model.
Uses a tiny synthetic dataset — no real CSV needed in CI.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

# ──────────────────────────────────────────────
# Synthetic data factory
# ──────────────────────────────────────────────

def make_synthetic_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    banks = ["BankA", "BankB", "BankC", "BankD"]
    accounts = [f"ACC{i:04d}" for i in range(100)]

    rows = []
    for i in range(n):
        from_bank = rng.choice(banks)
        to_bank   = rng.choice(banks)
        src_acc   = rng.choice(accounts)
        dst_acc   = rng.choice(accounts)
        rows.append({
            "Timestamp":          i,
            "From Bank":          from_bank,
            "Account":            src_acc,
            "To Bank":            to_bank,
            "Account.1":          dst_acc,
            "Amount Received":    float(rng.integers(100, 100000)),
            "Receiving Currency": "USD",
            "Amount Paid":        float(rng.integers(100, 100000)),
            "Payment Currency":   "USD",
            "Payment Format":     rng.choice(["Wire", "Cash", "Cheque"]),
            "Is Laundering":      int(rng.random() < 0.05),
            "src_id":             f"{from_bank}_{src_acc}",
            "dst_id":             f"{to_bank}_{dst_acc}",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Preprocessing tests
# ──────────────────────────────────────────────

class TestPreprocessing:
    def test_build_node_features(self):
        from src.preprocessing import build_node_features
        df    = make_synthetic_df(500)
        feats = build_node_features(df)
        assert "out_degree" in feats.columns
        assert "in_degree"  in feats.columns
        assert "total_sent" in feats.columns
        assert len(feats) > 0

    def test_build_node_labels(self):
        from src.preprocessing import build_node_features, build_node_labels
        df    = make_synthetic_df(500)
        feats = build_node_features(df).set_index("node_id")
        labels = build_node_labels(df, feats.index)
        assert set(labels.unique()) <= {0, 1}
        assert len(labels) == len(feats)

    def test_split_by_bank(self):
        from src.preprocessing import split_by_bank
        df = make_synthetic_df(1000)
        splits = split_by_bank(df, n_clients=4)
        assert len(splits) == 4
        total = sum(len(v) for v in splits.values())
        assert total <= len(df)


# ──────────────────────────────────────────────
# Graph building tests
# ──────────────────────────────────────────────

class TestGraphBuilder:
    def test_build_graph_shapes(self):
        from src.preprocessing import build_node_features, build_node_labels
        from src.graph_builder  import build_graph
        df    = make_synthetic_df(500)
        feats = build_node_features(df).set_index("node_id")
        labels = build_node_labels(df, feats.index)
        data, scaler, node_map = build_graph(df, feats, labels)

        assert data.x.ndim == 2
        assert data.x.shape[1] == 8
        assert data.y.ndim == 1
        assert data.edge_index.shape[0] == 2
        assert data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum() == data.num_nodes

    def test_masks_disjoint(self):
        from src.preprocessing import build_node_features, build_node_labels
        from src.graph_builder  import build_graph
        df    = make_synthetic_df(300)
        feats = build_node_features(df).set_index("node_id")
        labels = build_node_labels(df, feats.index)
        data, _, _ = build_graph(df, feats, labels)

        # Masks should be mutually exclusive
        assert (data.train_mask & data.val_mask).sum() == 0
        assert (data.train_mask & data.test_mask).sum() == 0
        assert (data.val_mask & data.test_mask).sum() == 0


# ──────────────────────────────────────────────
# Model tests
# ──────────────────────────────────────────────

class TestModel:
    def test_forward_pass(self):
        from src.model import GraphSAGE_AML
        model = GraphSAGE_AML(in_channels=8)
        model.eval()

        N = 100; E = 300
        x          = torch.randn(N, 8)
        edge_index = torch.randint(0, N, (2, E))

        with torch.no_grad():
            out = model(x, edge_index)

        assert out.shape == (N, 2), f"Expected ({N}, 2), got {out.shape}"
        assert not torch.isnan(out).any()

    def test_parameter_count(self):
        from src.model import GraphSAGE_AML
        model = GraphSAGE_AML(in_channels=8)
        # Should have reasonable parameter count (not too tiny, not too huge)
        n = model.count_parameters()
        assert 1_000 < n < 5_000_000, f"Unexpected parameter count: {n}"

    def test_embeddings(self):
        from src.model import GraphSAGE_AML
        model = GraphSAGE_AML(in_channels=8)
        model.eval()
        x = torch.randn(50, 8)
        edge_index = torch.randint(0, 50, (2, 100))
        with torch.no_grad():
            emb = model.get_embeddings(x, edge_index)
        assert emb.shape[0] == 50


# ──────────────────────────────────────────────
# Training tests (short smoke test)
# ──────────────────────────────────────────────

class TestTraining:
    def test_train_epoch(self):
        from src.preprocessing import build_node_features, build_node_labels
        from src.graph_builder  import build_graph
        from src.model          import GraphSAGE_AML
        from src.train          import train_epoch, compute_class_weights

        df    = make_synthetic_df(300)
        feats = build_node_features(df).set_index("node_id")
        labels = build_node_labels(df, feats.index)
        data, _, _ = build_graph(df, feats, labels)

        device = torch.device("cpu")
        model  = GraphSAGE_AML(in_channels=8).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=0.01)
        weights = compute_class_weights(data.y, device)

        loss = train_epoch(model, data, opt, weights, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate(self):
        from src.preprocessing import build_node_features, build_node_labels
        from src.graph_builder  import build_graph
        from src.model          import GraphSAGE_AML
        from src.train          import evaluate

        df    = make_synthetic_df(300)
        feats = build_node_features(df).set_index("node_id")
        labels = build_node_labels(df, feats.index)
        data, _, _ = build_graph(df, feats, labels)

        device = torch.device("cpu")
        model  = GraphSAGE_AML(in_channels=8).to(device)

        metrics = evaluate(model, data, data.val_mask, device)
        assert "accuracy" in metrics
        assert "f1"       in metrics
        assert "auc"      in metrics
        assert 0 <= metrics["accuracy"] <= 1
