"""Runtime tripwires. Fail loud and early — not 40 minutes into a FLAML run.
These are the cheap asserts that catch a bad target join or wrong file."""
from __future__ import annotations

import pandas as pd


def assert_training_contract(X: pd.DataFrame, y: pd.Series, cfg) -> None:
    # single-timestamp month => one row per CIF => within-month CV is clean.
    assert X[cfg.id_col].is_unique, f"{cfg.id_col} not unique in training month"
    assert len(X) > 0, "empty training frame"
    assert len(X) == len(y), "X / y length mismatch after target join"
    assert y.notna().any(), "target is all-null (bad join?)"
    assert y.nunique() > 1, "target is constant"

    base = float(y.mean())
    assert cfg.min_base_rate <= base <= cfg.max_base_rate, (
        f"base rate {base:.4f} outside sane band "
        f"[{cfg.min_base_rate}, {cfg.max_base_rate}] — check target build"
    )
