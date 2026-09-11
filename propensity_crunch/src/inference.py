"""Batch inference: score eligible clients at an inference date using persisted
trained models, emitting long-format rows.

Decoupled from training and reproducible: each product is scored with its own
FROZEN config + saved model + the calibration offset recorded in its card, so
scoring is independent of any later YAML edits. Only the inference date is a
free parameter (you retrain periodically, then score forward).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from configs._schema import RunConfig
from src.calibration import from_offset
from src.io import load_snapshot


def discover_products(art_root="artifacts"):
    """Products that have at least one runnable trained model."""
    root = Path(art_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and _latest_run(p) is not None)


def resolve_run(product, art_root="artifacts", run_id=None) -> Path:
    """Latest run for a product (default) or a pinned run_id."""
    pdir = Path(art_root) / product
    run_dir = (pdir / run_id) if run_id else _latest_run(pdir)
    if run_dir is None or not (run_dir / "model.pkl").exists():
        raise FileNotFoundError(f"no runnable model for {product!r} under {pdir}")
    return run_dir


def _latest_run(pdir: Path):
    if not pdir.exists():
        return None
    runs = [d for d in pdir.iterdir() if d.is_dir() and (d / "model.pkl").exists()]
    # run_id = <config_hash>_<YYYYmmdd-HHMMSS>; sort by the timestamp suffix
    return max(runs, key=lambda d: d.name.split("_")[-1], default=None)


def load_scorer(run_dir: Path):
    """Return (cfg, model, calibrate_or_None, model_hash) for a run directory."""
    cfg = RunConfig(**json.loads((run_dir / "config.frozen.json").read_text()))
    model_bytes = (run_dir / "model.pkl").read_bytes()
    model = pickle.loads(model_bytes)
    model_hash = hashlib.sha1(model_bytes).hexdigest()[:12]

    calibrate = None
    card_p = run_dir / "model_card.json"
    if card_p.exists():
        cal = json.loads(card_p.read_text()).get("calibration")
        if cal and cal.get("offset") is not None:
            calibrate = from_offset(cal["offset"])
    return cfg, model, calibrate, model_hash


def score_product(spark, cfg, model, calibrate, infer_date, model_hash) -> pd.DataFrame:
    """Score all eligible clients for one product at infer_date; long format:
    (cif, product, model_hash, infer_date, score)."""
    X = load_snapshot(spark, cfg, infer_date)          # eligible population + features
    p = model.predict_proba(X[cfg.features])[:, 1]
    if calibrate:
        p = calibrate(p)
    return pd.DataFrame(
        {
            cfg.id_col: X[cfg.id_col].values,
            "product": cfg.product,
            "model_hash": model_hash,
            "infer_date": str(infer_date),
            "score": p,
        }
    )
