"""External OOT evaluation against the declared population-base-rate baseline.
Run OUTSIDE the engine — FLAML does not grade its own OOT."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def oot_evaluate(y_true: pd.Series, scores: np.ndarray, metric: str) -> dict:
    y = np.asarray(y_true)
    base_rate = float(np.mean(y))
    return {
        "metric": metric,
        "oot_auc": float(roc_auc_score(y, scores)),
        "baseline_base_rate": base_rate,          # the declared baseline
        "top_decile_lift": _top_decile_lift(y, scores, base_rate),
        "n_oot": int(len(y)),
    }


def _top_decile_lift(y, s, base_rate) -> float:
    if base_rate == 0:
        return float("nan")
    k = max(1, int(0.10 * len(s)))
    top = np.argsort(s)[::-1][:k]
    return float(np.mean(y[top]) / base_rate)
