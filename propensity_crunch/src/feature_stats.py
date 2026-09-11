"""EDA / Feature-Checks statistics.

- vif / mutual_info      : multicollinearity and non-linear target dependence.
- univariate_summary     : compact per-feature quantile + null table (data dict).
- event_rate_by_category : count + target rate per categorical level.
- signal_persistence     : univariate AUC obs-vs-OOT + PSI drift (a *check*).
None is a correctness gate on its own; they inform the human decisions in
Feature Checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def _numeric(X, features):
    cols = list(features) if features is not None else X.select_dtypes("number").columns.tolist()
    M = X[cols].select_dtypes("number")
    return M.fillna(M.median(numeric_only=True))


def vif(X, features=None):
    """Variance inflation factor per feature = diagonal of the inverse
    correlation matrix. Rule of thumb: >5 notable, >10 serious multicollinearity."""
    M = _numeric(X, features)
    nz = M.std()
    M = M[nz[nz > 0].index]                     # drop zero-variance columns
    corr = np.atleast_2d(np.corrcoef(M.values, rowvar=False))
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)              # singular -> pseudo-inverse
    return pd.Series(np.diag(inv), index=M.columns, name="VIF").sort_values(ascending=False)


def mutual_info(X, y, features=None, seed=42):
    """Mutual information between each numeric feature and the target."""
    M = _numeric(X, features)
    mi = mutual_info_classif(M.values, np.asarray(y), random_state=seed)
    return pd.Series(mi, index=M.columns, name="MI").sort_values(ascending=False)


def univariate_summary(X, features=None):
    """Compact per-feature table: %null, min, p1, median, p99, max. Doubles as a
    quick data dictionary and an outlier scan (p99/max gap flags heavy tails)."""
    cols = list(features) if features is not None else X.columns.tolist()
    num = X[cols].select_dtypes("number")
    q = num.quantile([0.01, 0.5, 0.99])
    return pd.DataFrame({
        "pct_null": X[cols].isna().mean().reindex(num.columns).round(4),
        "min": num.min(), "p1": q.loc[0.01], "median": q.loc[0.50],
        "p99": q.loc[0.99], "max": num.max(),
    }).sort_values("pct_null", ascending=False)


def event_rate_by_category(X, y, col, top=20):
    """Count + target rate per level of a categorical feature. Surfaces
    predictive levels and rare levels that will destabilise CV. The dashed line
    to compare against is the overall base rate = float(y.mean())."""
    d = pd.DataFrame({col: X[col].astype("object"), "_y": np.asarray(y)})
    g = d.groupby(col, dropna=False)["_y"].agg(count="size", target_rate="mean")
    return g.sort_values("count", ascending=False).head(top)


def _univariate_auc(x, y):
    from sklearn.metrics import roc_auc_score
    x = pd.Series(x)
    if not pd.api.types.is_numeric_dtype(x) or x.notna().sum() == 0:
        return np.nan
    x = x.fillna(x.median())
    try:
        a = roc_auc_score(np.asarray(y), x.values)
        return max(a, 1 - a)                    # direction-agnostic
    except ValueError:
        return np.nan


def _psi(expected, actual, bins=10):
    """Population Stability Index of a numeric feature between two samples, using
    quantile bins fixed on the expected (train) distribution."""
    e = pd.Series(expected).dropna().values
    a = pd.Series(actual).dropna().values
    if len(e) == 0 or len(a) == 0:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan                            # not enough distinct values to bin
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(e, edges)[0] / len(e), 1e-6, None)
    a_pct = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def signal_persistence(X_obs, y_obs, X_oot, y_oot, features=None):
    """Per NUMERIC feature: univariate AUC on the train (obs) month vs the OOT
    month, their gap, and the PSI (input drift) between the two months. Flags
    features whose signal does not persist (auc_drop > 0.03) or that drift
    heavily (psi > 0.25) — the failure mode a single-month, in-sample screen
    cannot see. Categorical features are skipped (no numeric PSI). Sorted so the
    biggest concerns are on top."""
    cols = list(features) if features is not None else \
        X_obs.select_dtypes("number").columns.tolist()
    cols = [c for c in cols
            if c in X_oot.columns and pd.api.types.is_numeric_dtype(X_obs[c])]
    rows = []
    for f in cols:
        a_obs = _univariate_auc(X_obs[f], y_obs)
        a_oot = _univariate_auc(X_oot[f], y_oot)
        drop = (a_obs - a_oot) if (pd.notna(a_obs) and pd.notna(a_oot)) else np.nan
        psi = _psi(X_obs[f], X_oot[f])
        rows.append({
            "feature": f, "auc_obs": a_obs, "auc_oot": a_oot, "auc_drop": drop,
            "psi": psi,
            "flag": bool((pd.notna(drop) and drop > 0.03) or (pd.notna(psi) and psi > 0.25)),
        })
    return (pd.DataFrame(rows)
            .sort_values(["flag", "auc_drop"], ascending=[False, False])
            .reset_index(drop=True))
