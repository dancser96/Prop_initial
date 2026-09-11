"""External OOT evaluation. Given a large retail base, the operational metrics
are precision/recall/lift at the top of the score ranking (who you'd actually
contact), alongside AUC. Baseline is the population base rate. Run OUTSIDE the
engine — FLAML does not grade its own OOT.

All metrics here are rank-based, so they are unaffected by score calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def oot_evaluate(y_true, scores, metric) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(scores)
    base = float(y.mean())
    out = {
        "metric": metric,
        "oot_auc": float(roc_auc_score(y, s)),
        "baseline_base_rate": base,
        "n_oot": int(len(y)),
        "top_1pct": _at_k(y, s, 0.01, base),
        "top_10pct": _at_k(y, s, 0.10, base),
    }
    return out


def _at_k(y, s, k, base) -> dict:
    n = len(s)
    m = max(1, int(k * n))
    idx = np.argsort(s)[::-1][:m]            # top-m by score
    tp = float(y[idx].sum())
    pos = float(y.sum())
    precision = tp / m
    return {
        "n": int(m),
        "precision": float(precision),
        "recall": float(tp / pos) if pos > 0 else float("nan"),
        "lift": float(precision / base) if base > 0 else float("nan"),
    }
