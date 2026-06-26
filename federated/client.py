import logging
import numpy as np
import torch
import flwr as fl
from collections import OrderedDict
from torch_geometric.data import Data

from src.model import GraphSAGE_AML
from src.train import local_train, evaluate

log = logging.getLogger(__name__)


class AMLFederatedClient(fl.client.NumPyClient):

    def __init__(
        self,
        bank_name: str,
        data: Data,
        in_channels: int,
        local_epochs: int = 5,
        device: str = None,
    ):
        self.bank_name = bank_name
        self.data = data
        self.local_epochs = local_epochs

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ⚠️ MUST MATCH SERVER MODEL EXACTLY
        self.model = GraphSAGE_AML(in_channels=in_channels).to(self.device)

        log.info(f"[{bank_name}] client ready")

    # ─────────────── FL methods ───────────────

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state = {}

        for k, v in zip(state_dict.keys(), parameters):
            new_state[k] = torch.tensor(v)

        # 🔥 CRITICAL FIX (avoid crash on mismatch)
        self.model.load_state_dict(new_state, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = float(config.get("lr", 0.005))
        epochs = int(config.get("local_epochs", self.local_epochs))

        params, num_examples, metrics = local_train(
            model=self.model,
            data=self.data,
            epochs=epochs,
            lr=lr,
            device=str(self.device),
        )

        return params, num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Pass data + device; restrict metrics to test split via mask
        metrics = evaluate(
            self.model,
            self.data,
            self.device,
            mask=self.data.test_mask,
        )

        return float(1.0 - metrics["f1"]), int(self.data.test_mask.sum()), metrics


# factory
def start_client(bank_name, data, in_channels, server_addr="127.0.0.1:8080"):
    client = AMLFederatedClient(
        bank_name=bank_name,
        data=data,
        in_channels=in_channels,
    )

    fl.client.start_numpy_client(server_address=server_addr, client=client)