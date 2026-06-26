"""
src/mlflow_tracker.py
Central MLflow logging utility used by both centralized and federated training.

What gets tracked:
  • Hyperparameters (lr, epochs, dataset size, model architecture)
  • Per-epoch metrics (loss, f1, auc, accuracy, precision, recall)
  • Per-round federated metrics
  • Final test results
  • Model artifact (.pt file)
  • Training plots (PNG)
  • System tags (GPU name, CUDA version)

View results:
  mlflow ui
  → open http://127.0.0.1:5000
"""

import os
import logging
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ── MLflow tracking URI ───────────────────────────────────────────
# Default: local ./mlruns folder.
# Change to a remote server: "http://your-mlflow-server:5000"
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)


def setup_mlflow(experiment_name: str = "AML_Detection") -> str:
    """
    Set tracking URI and create (or reuse) the experiment.
    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    log.info(f"MLflow → tracking URI : {TRACKING_URI}")
    log.info(f"MLflow → experiment   : {experiment_name}  (id={exp.experiment_id})")
    return exp.experiment_id


# ══════════════════════════════════════════════════════════════════
# Centralized training tracker
# ══════════════════════════════════════════════════════════════════

class CentralizedTracker:
    """
    Context-manager wrapper around an MLflow run for centralized training.

    Usage:
        with CentralizedTracker(params) as tracker:
            for epoch in range(epochs):
                loss = train(...)
                metrics = evaluate(...)
                tracker.log_epoch(epoch, loss, metrics)
            tracker.log_final(test_metrics, model, scaler)
    """

    def __init__(self, params: Dict, run_name: str = "centralized"):
        self.params   = params
        self.run_name = run_name
        self.run      = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        self._log_system_tags()
        mlflow.log_params(self.params)
        log.info(f"MLflow run started: {self.run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()
        log.info("MLflow run ended.")

    def log_epoch(self, epoch: int, loss: float, train_m: Dict, val_m: Dict):
        mlflow.log_metrics({
            "train/loss":      loss,
            "train/f1":        train_m.get("f1", 0),
            "train/accuracy":  train_m.get("accuracy", 0),
            "val/f1":          val_m.get("f1", 0),
            "val/accuracy":    val_m.get("accuracy", 0),
            "val/precision":   val_m.get("precision", 0),
            "val/recall":      val_m.get("recall", 0),
            "val/auc":         val_m.get("auc", 0),
        }, step=epoch)

    def log_final(
        self,
        test_metrics: Dict,
        model,
        scaler=None,
        model_path: Optional[str] = None,
        plot_path:  Optional[str] = None,
    ):
        # Final test metrics with "test/" prefix
        mlflow.log_metrics({
            f"test/{k}": v for k, v in test_metrics.items()
        })
        mlflow.set_tag("status", "SUCCESS")

        # Log model file as artifact
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_path="model")

        # Log training plot
        if plot_path and Path(plot_path).exists():
            mlflow.log_artifact(plot_path, artifact_path="plots")

        # Log PyTorch model directly (enables mlflow models serve)
        try:
            mlflow.pytorch.log_model(model, "pytorch_model")
        except Exception as e:
            log.warning(f"Could not log pytorch model: {e}")

    def _log_system_tags(self):
        mlflow.set_tag("python_version", f"{__import__('sys').version.split()[0]}")
        if torch.cuda.is_available():
            mlflow.set_tag("gpu",  torch.cuda.get_device_name(0))
            mlflow.set_tag("cuda", torch.version.cuda)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            mlflow.set_tag("vram_gb", str(vram))
        else:
            mlflow.set_tag("gpu", "cpu")


# ══════════════════════════════════════════════════════════════════
# Federated training tracker
# ══════════════════════════════════════════════════════════════════

class FederatedTracker:
    """
    Tracks federated learning across rounds and per-client metrics.

    Usage:
        with FederatedTracker(params) as tracker:
            for round_num in range(1, rounds+1):
                # after each FL round:
                tracker.log_round(round_num, client_metrics_list)
            tracker.log_final(avg_test_metrics, model_path)
    """

    def __init__(self, params: Dict, run_name: str = "federated"):
        self.params   = params
        self.run_name = run_name
        self.run      = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        self._log_system_tags()
        mlflow.log_params(self.params)
        log.info(f"MLflow federated run: {self.run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()

    def log_round(self, round_num: int, client_results: list):
        """
        client_results: list of dicts with keys f1, auc, loss, accuracy, etc.
        Logs the weighted average across clients.
        """
        if not client_results:
            return

        n = len(client_results)
        avg = {
            k: sum(r.get(k, 0) for r in client_results) / n
            for k in ["f1", "auc", "accuracy", "precision", "recall", "loss"]
        }
        mlflow.log_metrics({
            f"federated/{k}": v for k, v in avg.items()
        }, step=round_num)

        # Per-client metrics as separate tags (for the last round)
        for i, r in enumerate(client_results):
            mlflow.log_metric(f"client_{i}/f1",  r.get("f1", 0),  step=round_num)
            mlflow.log_metric(f"client_{i}/auc", r.get("auc", 0), step=round_num)

        log.info(
            f"  [MLflow] Round {round_num:02d} → "
            f"avg F1:{avg['f1']:.4f}  AUC:{avg['auc']:.4f}"
        )

    def log_final(self, test_metrics: Dict, model_path: Optional[str] = None):
        mlflow.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})
        mlflow.set_tag("status", "SUCCESS")
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_path="model")

    def _log_system_tags(self):
        mlflow.set_tag("python_version", f"{__import__('sys').version.split()[0]}")
        mlflow.set_tag("framework", "Flower FedAvg")
        if torch.cuda.is_available():
            mlflow.set_tag("gpu",  torch.cuda.get_device_name(0))
            mlflow.set_tag("cuda", torch.version.cuda)
        else:
            mlflow.set_tag("gpu", "cpu")


# ── Convenience: log a dict of params + metrics in one shot ───────

def quick_log(run_name: str, params: Dict, metrics: Dict, tags: Dict = None):
    """One-liner for simple logging without a context manager."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))
