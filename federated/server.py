import logging
import flwr as fl
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)


class AMLFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            total = sum(r.num_examples for _, r in results)

            f1 = sum(
                r.metrics.get("f1", 0) * r.num_examples
                for _, r in results
            ) / total

            log.info(f"[Round {server_round}] F1={f1:.4f}")

        return agg_params, agg_metrics


def build_strategy(num_clients: int):

    return AMLFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,

        on_fit_config_fn=lambda r: {
            "local_epochs": 5,
            "lr": max(0.001, 0.005 * (0.98 ** r)),
        },
    )