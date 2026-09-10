"""Minimal auto-generated card: frozen intent + OOT metrics + feature list +
caveats, plus a best-effort importance plot. Falls out of the run for free."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_card(cfg, metrics: dict, feats: list[str], automl, out: Path) -> None:
    _importance_plot(automl, feats, out)

    card = {
        "product": cfg.product,
        "run_dates": {
            "obs": str(cfg.obs_date),
            "oot": str(cfg.oot_date),
            "infer": str(cfg.infer_date),
        },
        "metric_declared_before_run": cfg.metric,
        "baseline": cfg.baseline,
        "oot_metrics": metrics,
        "n_features": len(feats),
        "best_estimator": getattr(automl, "best_estimator", None),
        "caveats": [
            "Single-timestamp training bakes in that month's seasonality "
            "(March 2026 overlaps Ramadan/Eid in the UAE — activation base "
            "rates may be atypical). Do not over-trust the level.",
            "Leakage screen is first-order only; second-order/temporal leakage "
            "is caught, if at all, by the OOT split.",
            "No automated feature selection — curated candidate set fed to the "
            "engine directly.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str))


def _importance_plot(automl, feats, out: Path) -> None:
    # Engine-dependent; best-effort only.
    try:
        imp = list(getattr(automl, "feature_importances_", []) or [])
        if not imp:
            imp = list(getattr(getattr(automl, "model", None), "feature_importances_", []) or [])
        if not imp or len(imp) != len(feats):
            return
        order = sorted(range(len(imp)), key=lambda i: imp[i], reverse=True)[:20]
        plt.figure(figsize=(6, 6))
        plt.barh([feats[i] for i in order][::-1], [imp[i] for i in order][::-1])
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.savefig(out / "feature_importance.png", dpi=110)
        plt.close()
    except Exception:
        pass
