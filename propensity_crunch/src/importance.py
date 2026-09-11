"""Model-agnostic feature importance: permutation importance on the OOT set.

Works for ANY fitted model exposing predict_proba (all FLAML estimators, and
anything added later) — it measures the OOT-AUC drop when a feature's values are
shuffled, so it needs no knowledge of the model family. OOT is subsampled for
speed; the raw drop in AUC is the importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def permutation_importance(model, X, y, features, n_repeats=3, seed=42, max_rows=50_000):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if len(X) > max_rows:
        take = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[take], y[take]
    Xf = X[features].reset_index(drop=True).copy()

    base = roc_auc_score(y, model.predict_proba(Xf)[:, 1])
    rows = []
    for f in features:
        orig = Xf[f].to_numpy(copy=True)
        drops = []
        for _ in range(n_repeats):
            Xf[f] = rng.permutation(orig)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xf)[:, 1]))
        Xf[f] = orig
        rows.append({"feature": f, "importance": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return (pd.DataFrame(rows)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
