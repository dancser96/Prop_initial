"""First-order leakage screen. FLAGS for human review — never auto-drops.
A legitimately strong feature and a leaking one look identical here; a human
decides which goes into features_exclude."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def leakage_report(X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.DataFrame:
    yv = y.values
    rows = []
    for f in features:
        s = X[f]
        miss = float(s.isna().mean())
        vc = s.value_counts(dropna=True, normalize=True)
        dominance = float(vc.iloc[0]) if len(vc) else 1.0

        auc = np.nan
        if pd.api.types.is_numeric_dtype(s) and s.notna().sum() > 0:
            x = s.fillna(s.median())
            try:
                a = roc_auc_score(yv, x)
                auc = max(a, 1 - a)            # direction-agnostic separability
            except ValueError:
                auc = np.nan

        rows.append(
            {
                "feature": f,
                "univariate_auc": auc,
                "pct_missing": miss,
                "value_dominance": dominance,
                "suspected_leak": bool(pd.notna(auc) and auc > 0.95),
            }
        )

    rep = pd.DataFrame(rows).sort_values(
        "univariate_auc", ascending=False, na_position="last"
    )
    return rep.reset_index(drop=True)
