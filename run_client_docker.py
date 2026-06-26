"""
run_client_docker.py
Entry point for each Flower bank client when running in Docker.

Reads environment variables:
  BANK_ID     : 0, 1, 2, or 3  (index into sorted bank list)
  SERVER_ADDR : host:port of the Flower server
  NROWS       : max rows to load from CSV
  TRANS_PATH  : path to *_Trans.csv inside container
  ACCOUNTS_PATH: path to *_accounts.csv inside container
"""

import os
import logging
import torch

from src.preprocessing import full_pipeline
from src.graph_builder  import build_graph, build_bank_graph
from federated.client   import start_client

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [CLIENT] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    bank_id      = int(os.getenv("BANK_ID",      "0"))
    server_addr  = os.getenv("SERVER_ADDR",       "flower-server:8080")
    nrows        = int(os.getenv("NROWS",         "500000"))
    trans_path   = os.getenv("TRANS_PATH",        "data/HI-Medium_Trans.csv")
    accounts_path= os.getenv("ACCOUNTS_PATH",     "data/HI-Medium_accounts.csv")
    n_clients    = int(os.getenv("N_CLIENTS",     "4"))
    device_str   = "cpu"   # CPU in Docker (override for GPU containers)

    log.info(f"Client BANK_ID={bank_id}  server={server_addr}  nrows={nrows}")

    df, node_feats, node_labels, bank_splits = full_pipeline(
        trans_path    = trans_path,
        accounts_path = accounts_path,
        nrows         = nrows,
        n_clients     = n_clients,
    )

    # Build global scaler from full data
    _, scaler, _ = build_graph(df, node_feats, node_labels)

    bank_names = list(bank_splits.keys())
    if bank_id >= len(bank_names):
        log.error(f"BANK_ID={bank_id} but only {len(bank_names)} banks found.")
        return

    bank_name = bank_names[bank_id]
    bank_df   = bank_splits[bank_name]
    bank_data = build_bank_graph(bank_df, node_feats, node_labels, scaler)

    log.info(f"Bank: {bank_name}  graph: {bank_data}")

    start_client(
        bank_name   = bank_name,
        data        = bank_data,
        server_addr = server_addr,
        device_str  = device_str,
        in_channels = bank_data.x.shape[1],
    )


if __name__ == "__main__":
    main()
