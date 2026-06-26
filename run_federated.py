import argparse
import logging
import os
import numpy as np
import torch
import flwr as fl

from src.preprocessing import full_pipeline
from src.graph_builder import build_graph, build_bank_graph
from src.model import GraphSAGE_AML
from src.train import evaluate
from src.mlflow_tracker import setup_mlflow, FederatedTracker
from federated.client import AMLFederatedClient
from federated.server import build_strategy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--trans", default="data/HI-Medium_Trans.csv")
    p.add_argument("--accounts", default="data/HI-Medium_accounts.csv")
    p.add_argument("--nrows", type=int, default=None)

    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_mlflow("AML")

    # ── data
    df, node_feats, node_labels, bank_splits = full_pipeline(
        args.trans,
        args.accounts,
        nrows=args.nrows,
        n_clients=args.clients,
    )

    global_data, scaler, _ = build_graph(df, node_feats, node_labels)
    in_channels = global_data.x.shape[1]

    bank_names = list(bank_splits.keys())

    bank_data = {
        n: build_bank_graph(bdf, node_feats, node_labels, scaler)
        for n, bdf in bank_splits.items()
    }

    bank_list = [(n, bank_data[n]) for n in bank_names]

    # ── client factory
    def client_fn(cid: str):
        name, data = bank_list[int(cid)]

        client = AMLFederatedClient(
            bank_name=name,
            data=data,
            in_channels=in_channels,
            local_epochs=args.epochs,
            device=device,
        )

        return client.to_client()

    # ── strategy
    strategy = build_strategy(args.clients)

    # ── run
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    log.info("Training complete")


if __name__ == "__main__":
    main()