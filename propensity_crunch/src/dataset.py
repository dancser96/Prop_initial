"""Labelled-frame builder + train-only downsampling, shared by run.py and the
notebooks so this logic lives in one place."""
from __future__ import annotations

import numpy as np

from src.io import load_snapshot


def load_labelled(spark, cfg, month, feature_cols=None):
    """Return (X, y): pruned feature frame + 0/1 label from the materialised
    target column. `feature_cols` picks the projection (default cfg.features;
    pass cfg.candidates for exploration). Null-target rows are dropped."""
    pdf = load_snapshot(spark, cfg, month, extra_cols=[cfg.target_col],
                        feature_cols=feature_cols)
    pdf = pdf[pdf[cfg.target_col].notna()].reset_index(drop=True)
    y = pdf[cfg.target_col].astype(int)
    X = pdf.drop(columns=[cfg.target_col])
    return X, y


def downsample_train(X, y, spec):
    """TRAIN ONLY. Keep all positives, sample negatives to `neg_per_pos` per
    positive. Shifts the base rate (scores need calibration) but preserves
    ranking. Never call on OOT or inference data."""
    pos = np.where(y.values == 1)[0]
    neg = np.where(y.values == 0)[0]
    n_keep = min(len(neg), spec.neg_per_pos * len(pos))
    keep_neg = np.random.default_rng(spec.seed).choice(neg, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([pos, keep_neg]))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
