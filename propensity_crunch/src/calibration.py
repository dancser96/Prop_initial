"""Score calibration, applied automatically when the training frame was
downsampled. One method now (prior_shift); registry keeps it swappable.

prior_shift: downsampling keeps all positives and a fraction of negatives, which
inflates the prior odds. The exact analytic correction shifts the log-odds by
logit(true_rate) - logit(sampled_rate), mapping probabilities back to the true
base rate. No held-out set needed; ranking/AUC are unaffected.
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def prior_shift(true_rate, sampled_rate):
    offset = float(_logit(true_rate) - _logit(sampled_rate))

    def apply(p):
        return _sigmoid(_logit(p) + offset)

    apply.method = "prior_shift"
    apply.offset = offset
    return apply


def from_offset(offset):
    """Rebuild a calibrator from a stored log-odds offset (used at inference time,
    where only the offset recorded in the card is available)."""
    def apply(p):
        return _sigmoid(_logit(p) + float(offset))

    apply.method = "offset"
    apply.offset = float(offset)
    return apply


CALIBRATORS = {"prior_shift": prior_shift}


def get_calibrator(method):
    if method not in CALIBRATORS:
        raise ValueError(f"unknown calibration {method!r}; have {sorted(CALIBRATORS)}")
    return CALIBRATORS[method]
