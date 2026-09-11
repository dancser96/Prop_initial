"""Minimal auto-generated card: frozen intent + OOT metrics + settings +
model-agnostic importance + caveats. Falls out of the run for free."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_card(cfg, metrics, feats, imp, out: Path, calibration=None, model_info=None) -> None:
    _importance_plot(imp, out)

    caveats = [
        "Single-month training bakes in that month's seasonality (March 2026 "
        "overlaps Ramadan/Eid in the UAE — base rates may be atypical).",
        "Leakage screen is first-order only; second-order/temporal leakage is "
        "caught, if at all, by the OOT split.",
        "No automated feature selection — curated features fed to the engine directly.",
    ]
    if calibration:
        caveats.append(
            f"Training was downsampled, so scores were calibrated back to the true base "
            f"rate via {calibration['method']} (log-odds offset {calibration['offset']:+.3f}). "
            "Rank-based OOT metrics are unaffected."
        )

    card = {
        "product": cfg.product,
        "run_dates": {"obs": str(cfg.obs_date), "oot": str(cfg.oot_date),
                      "infer": str(cfg.infer_date)},
        "metric_declared_before_run": cfg.metric,
        "baseline": cfg.baseline,
        "oot_metrics": metrics,
        "n_features": len(feats),
        "eligibility": cfg.eligibility_expr,
        "downsample": cfg.downsample.model_dump() if cfg.downsample else None,
        "calibration": calibration,
        "model_search": cfg.model.model_dump(),   # what the search was allowed to do
        "model_selected": model_info,             # what it actually picked (best_config etc.)
        "top_importance": imp.head(10).to_dict("records"),
        "caveats": caveats,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str))


def _importance_plot(imp, out: Path) -> None:
    try:
        top = imp.head(20).iloc[::-1]
        plt.figure(figsize=(6, 6))
        plt.barh(top["feature"], top["importance"])
        plt.title("Permutation importance (OOT AUC drop)")
        plt.tight_layout()
        plt.savefig(out / "feature_importance.png", dpi=110)
        plt.close()
    except Exception:
        pass
