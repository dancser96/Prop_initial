"""SEAM 2 — the run-harness. config-in, artifacts-out, callable.

Correctness checks WRAP the engine: the leakage screen, permutation importance,
and OOT evaluation are external; FLAML sees only the curated feature set and
never grades its own OOT.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from flaml import AutoML

from src.calibration import get_calibrator
from src.card import write_card
from src.contract import assert_training_contract
from src.dataset import downsample_train, load_labelled
from src.evaluate import oot_evaluate
from src.importance import permutation_importance
from src.io import load_snapshot
from src.leakage import leakage_report

ART_ROOT = Path("artifacts")


def run(cfg, spark) -> dict:
    run_id = _run_id(cfg)
    out = ART_ROOT / cfg.product / run_id
    out.mkdir(parents=True, exist_ok=True)
    _freeze_config(cfg, out)

    feats = cfg.features

    # --- TRAIN month -------------------------------------------------------
    Xtr, ytr = load_labelled(spark, cfg, cfg.obs_date)
    assert_training_contract(Xtr, ytr, cfg)
    leakage_report(Xtr, ytr, feats).to_csv(out / "leakage_report.csv", index=False)

    true_rate = float(ytr.mean())
    calibrate = None
    sampled_rate = true_rate
    if cfg.downsample:                                  # train-only; OOT/inference untouched
        Xtr, ytr = downsample_train(Xtr, ytr, cfg.downsample)
        sampled_rate = float(ytr.mean())
        calibrate = get_calibrator(cfg.calibration)(true_rate, sampled_rate)

    # --- fit (curated features only; NO auto feature-selection) -----------
    automl = AutoML()
    automl.fit(
        X_train=Xtr[feats], y_train=ytr, task="classification", metric=cfg.metric,
        time_budget=cfg.model.time_budget, estimator_list=cfg.model.estimator_list,
        eval_method=cfg.model.eval_method, n_splits=cfg.model.n_splits,
        ensemble=cfg.model.ensemble, sample=cfg.model.sample,
        early_stop=cfg.model.early_stop, n_jobs=cfg.model.n_jobs, seed=cfg.model.seed,
        log_file_name=str(out / "flaml_search.log"), verbose=1,
    )

    # --- OOT evaluation + model-agnostic permutation importance -----------
    Xoot, yoot = load_labelled(spark, cfg, cfg.oot_date)
    soot = automl.predict_proba(Xoot[feats])[:, 1]
    metrics = oot_evaluate(yoot, soot, cfg.metric)
    (out / "oot_metrics.json").write_text(json.dumps(metrics, indent=2))

    imp = permutation_importance(automl, Xoot, yoot, feats)
    imp.to_csv(out / "permutation_importance.csv", index=False)

    # --- inference + calibrated score log ---------------------------------
    Xinf = load_snapshot(spark, cfg, cfg.infer_date)
    raw = automl.predict_proba(Xinf[feats])[:, 1]
    score = calibrate(raw) if calibrate else raw
    pd.DataFrame(
        {
            cfg.id_col: Xinf[cfg.id_col].values,
            "score": score, "score_raw": raw,
            "calibrated": bool(calibrate is not None),
            "config_hash": run_id.split("_")[0],
            "feature_set_signature": _feature_sig(feats),
            "infer_date": str(cfg.infer_date), "run_id": run_id,
        }
    ).to_parquet(out / "scores.parquet", index=False)

    # --- persist model + card ---------------------------------------------
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(automl, fh)
    calib_info = (
        {"method": calibrate.method, "offset": calibrate.offset,
         "true_rate": true_rate, "sampled_rate": sampled_rate}
        if calibrate else None
    )
    model_info = {
        "best_estimator": getattr(automl, "best_estimator", None),
        "best_config": getattr(automl, "best_config", None),      # winning hyperparameters
        "best_cv_loss": getattr(automl, "best_loss", None),       # best CV score (lower = better)
    }
    write_card(cfg, metrics, feats, imp, out, calibration=calib_info, model_info=model_info)

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
