"""
api/main.py
FastAPI inference server.
Loads the trained model and exposes:
  GET  /health
  POST /predict          — single account risk score
  POST /predict/batch    — bulk account scoring
  GET  /stats            — dataset summary statistics
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import GraphSAGE_AML

log = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AML Risk Detection API",
    description = "Federated GraphSAGE — Anti-Money Laundering",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global state (loaded at startup) ─────────────────────────────────

MODEL_PATH  = os.getenv("MODEL_PATH", "outputs/centralized_model.pt")
STATS_PATH  = os.getenv("STATS_PATH", "outputs/centralized_results.json")

model: Optional[GraphSAGE_AML] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global model
    if not Path(MODEL_PATH).exists():
        log.warning(f"Model not found at {MODEL_PATH}. /predict will return 503.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    in_ch      = checkpoint.get("in_channels", 8)
    model      = GraphSAGE_AML(in_channels=in_ch).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info(f"Model loaded from {MODEL_PATH} on {device}")


# ── Request / Response schemas ────────────────────────────────────────

class AccountFeatures(BaseModel):
    account_id:     str
    out_degree:     float = 0
    in_degree:      float = 0
    total_sent:     float = 0
    total_received: float = 0
    avg_sent:       float = 0
    avg_received:   float = 0
    unique_out_peers: float = 0
    unique_in_peers:  float = 0


class RiskScore(BaseModel):
    account_id:   str
    risk_score:   float          # 0.0 – 1.0
    label:        int            # 0 normal / 1 laundering
    confidence:   float
    risk_level:   str            # LOW / MEDIUM / HIGH / CRITICAL


class BatchRequest(BaseModel):
    accounts: List[AccountFeatures]


def _feature_tensor(acc: AccountFeatures) -> torch.Tensor:
    """Pack the 8 features into a 2-D tensor (1, 8)."""
    import math
    feats = [
        acc.out_degree,
        acc.in_degree,
        math.log1p(acc.total_sent),
        math.log1p(acc.total_received),
        math.log1p(acc.avg_sent),
        math.log1p(acc.avg_received),
        acc.unique_out_peers,
        acc.unique_in_peers,
    ]
    return torch.tensor([feats], dtype=torch.float).to(device)


def _risk_level(score: float) -> str:
    if score < 0.25:   return "LOW"
    if score < 0.50:   return "MEDIUM"
    if score < 0.75:   return "HIGH"
    return "CRITICAL"


def _infer_single(acc: AccountFeatures) -> RiskScore:
    """Run model inference for one account (no graph structure — node only)."""
    x          = _feature_tensor(acc)
    edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)  # isolated node

    with torch.no_grad():
        logits = model(x, edge_index)
        probs  = torch.softmax(logits, dim=1)

    p_launder = float(probs[0, 1])
    label     = int(p_launder >= 0.5)

    return RiskScore(
        account_id = acc.account_id,
        risk_score = round(p_launder, 4),
        label      = label,
        confidence = round(float(probs[0, label]), 4),
        risk_level = _risk_level(p_launder),
    )


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "device": str(device),
        "model":  MODEL_PATH,
    }


@app.post("/predict", response_model=RiskScore)
def predict(account: AccountFeatures):
    if model is None:
        raise HTTPException(503, "Model not loaded. Train first with run_centralized.py")
    return _infer_single(account)


@app.post("/predict/batch", response_model=List[RiskScore])
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")
    return [_infer_single(acc) for acc in request.accounts]


@app.get("/stats")
def stats():
    if Path(STATS_PATH).exists():
        with open(STATS_PATH) as f:
            return json.load(f)
    return {"message": "No results found. Run training first."}


# ── Dev server ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
