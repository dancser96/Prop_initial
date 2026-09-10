"""SEAM 2 — the run-harness. config-in, artifacts-out, callable.

This is the prototype of the framework's execution surface. It enforces the
invariant that the correctness checks WRAP the engine: leakage screen and OOT
are external; FLAML sees only the curated feature set and never grades its own
OOT.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from flaml import AutoML

from src.card import write_card
from src.contract import assert_training_contract
from src.dataset import load_labelled
from src.evaluate import oot_evaluate
from src.io import load_snapshot
from src.leakage import leakage_report

ART_ROOT = Path("artifacts")


def run(cfg, spark) -> dict:
    run_id = _run_id(cfg)
    out = ART_ROOT / cfg.product / run_id
    out.mkdir(parents=True, exist_ok=True)
    _freeze_config(cfg, out)

    feats = cfg.features

    # --- TRAIN month (obs) ------------------------------------------------
    Xtr, ytr = load_labelled(spark, cfg, cfg.obs_date)
    assert_training_contract(Xtr, ytr, cfg)

    # leakage screen — WRITE for human review, do NOT auto-drop
    leakage_report(Xtr, ytr, feats).to_csv(out / "leakage_report.csv", index=False)

    # --- fit FLAML on curated features only (NO auto feature-selection) ---
    automl = AutoML()
    automl.fit(
        X_train=Xtr[feats],
        y_train=ytr,
        task="classification",
        metric=cfg.metric,
        time_budget=cfg.model.time_budget,
        estimator_list=cfg.model.estimator_list,
        eval_method="cv",
        n_splits=cfg.model.n_splits,
        seed=cfg.model.seed,
        verbose=1,
    )

    # --- OOT evaluation (external, later month) ---------------------------
    Xoot, yoot = load_labelled(spark, cfg, cfg.oot_date)
    soot = automl.predict_proba(Xoot[feats])[:, 1]
    metrics = oot_evaluate(yoot, soot, cfg.metric)
    (out / "oot_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- inference (infer month) + score log ------------------------------
    Xinf = load_snapshot(spark, cfg, cfg.infer_date)
    sinf = automl.predict_proba(Xinf[feats])[:, 1]
    pd.DataFrame(
        {
            cfg.id_col: Xinf[cfg.id_col].values,
            "score": sinf,
            "config_hash": run_id.split("_")[0],
            "feature_set_signature": _feature_sig(feats),
            "infer_date": str(cfg.infer_date),
            "run_id": run_id,
        }
    ).to_parquet(out / "scores.parquet", index=False)

    # --- persist model + card ---------------------------------------------
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(automl, fh)
    write_card(cfg, metrics, feats, automl, out)

    return {"run_id": run_id, "out": str(out), "metrics": metrics}


# --------------------------------------------------------------------------
def _run_id(cfg) -> str:
    return f"{_config_hash(cfg)}_{datetime.now():%Y%m%d-%H%M%S}"


def _config_hash(cfg) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _feature_sig(feats) -> str:
    return hashlib.sha1("|".join(sorted(feats)).encode()).hexdigest()[:10]


def _freeze_config(cfg, out: Path) -> None:
    (out / "config.frozen.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
    )
