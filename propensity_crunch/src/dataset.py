"""Labelled-frame builder shared by run.py AND the stage notebooks, so the
target read + alignment lives in exactly one place.

The target is a column materialised during the Data Creation stage; here it is
read alongside the features and split off."""
from __future__ import annotations

from src.io import load_snapshot


def load_labelled(spark, cfg, month):
    """Return (X, y): pruned feature frame + 0/1 label.

    Rows with a null target (e.g. panel-edge months without a full forward
    window) are dropped so they cannot leak into training or OOT.
    """
    pdf = load_snapshot(spark, cfg, month, extra_cols=[cfg.target_col])
    pdf = pdf[pdf[cfg.target_col].notna()].reset_index(drop=True)
    y = pdf[cfg.target_col].astype(int)
    X = pdf.drop(columns=[cfg.target_col])
    return X, y
