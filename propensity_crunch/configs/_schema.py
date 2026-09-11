"""SEAM 1 — the config spine: ONE schema every product config conforms to,
validated on load. Pydantic v2.

A config is one product's entire run definition. It exposes TWO feature views of
one file:
  - `candidates` = features_include            -> what EDA + Feature Checks read
  - `features`   = features_include - exclude   -> what the model reads
So exploration always sees the full candidate set (including currently-excluded
features), while modelling reads the post-decision survivors. One source of truth.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_METRICS = {"roc_auc", "ap", "log_loss", "f1", "accuracy", "macro_f1", "micro_f1"}
ALLOWED_EVAL = {"cv", "holdout", "auto"}
ALLOWED_CALIB = {"prior_shift"}


class ModelSpec(BaseModel):
    """FLAML search settings — the knobs worth exposing (see DOCS.md §5)."""
    time_budget: int = Field(300, ge=30)     # search seconds
    estimator_list: list[str] = ["lgbm", "xgboost", "rf", "extra_tree"]
    eval_method: str = "cv"                   # cv | holdout | auto
    n_splits: int = 5                         # within-month => clean folds
    ensemble: bool = False
    sample: bool = True                       # FLAML subsample during search (speed only)
    early_stop: bool = True
    n_jobs: int = -1
    seed: int = 42

    @field_validator("eval_method")
    @classmethod
    def _eval_ok(cls, v):
        if v not in ALLOWED_EVAL:
            raise ValueError(f"eval_method {v!r} not in {sorted(ALLOWED_EVAL)}")
        return v


class Downsample(BaseModel):
    """Optional, TRAIN-ONLY: keep all positives, sample negatives to
    `neg_per_pos` per positive. Auto-paired with calibration (see RunConfig)."""
    neg_per_pos: int = Field(ge=1)
    seed: int = 42


class RunConfig(BaseModel):
    product: str
    table: str                      # Hive table name OR HDFS parquet path
    id_col: str = "cif"
    # month_col: column stamping each row's monthly snapshot (one row per id per
    # month). Selects obs/OOT/inference months and anchors the rollups.
    month_col: str = "snapshot_month"

    obs_date: date                  # train / within-month CV
    oot_date: date                  # out-of-time validation
    infer_date: date                # scoring

    features_include: list[str]     # curated candidate set — NO automated selection
    features_exclude: list[str] = []   # human-decided drops from Feature Checks
    eligibility_expr: Optional[str] = None

    target_col: str
    window_months: int = Field(ge=1)

    model: ModelSpec = ModelSpec()
    downsample: Optional[Downsample] = None
    # Calibration method applied automatically WHEN downsampling is set (there is
    # no prior shift to correct otherwise). One method now, swappable later.
    calibration: str = "prior_shift"

    metric: str = "roc_auc"                     # declared BEFORE results
    baseline: str = "population_base_rate"
    min_base_rate: float = 0.001
    max_base_rate: float = 0.60

    @field_validator("features_include")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("features_include is empty")
        return v

    @field_validator("metric")
    @classmethod
    def _metric_ok(cls, v):
        if v not in ALLOWED_METRICS:
            raise ValueError(f"metric {v!r} not in {sorted(ALLOWED_METRICS)}")
        return v

    @field_validator("calibration")
    @classmethod
    def _calib_ok(cls, v):
        if v not in ALLOWED_CALIB:
            raise ValueError(f"calibration {v!r} not in {sorted(ALLOWED_CALIB)}")
        return v

    @model_validator(mode="after")
    def _temporal_safety(self):
        if not (self.obs_date < self.oot_date < self.infer_date):
            raise ValueError("require obs_date < oot_date < infer_date")
        gap = _month_diff(self.obs_date, self.oot_date)
        if gap < self.window_months:
            raise ValueError(
                f"oot_date must be >= window_months ({self.window_months}m) after "
                f"obs_date so train/OOT target windows do not overlap; got {gap}m"
            )
        return self

    @property
    def candidates(self) -> list[str]:
        """Pre-decision candidate set — what exploration reads."""
        return list(self.features_include)

    @property
    def features(self) -> list[str]:
        """Final model input = candidates minus human-decided exclusions."""
        excl = set(self.features_exclude)
        return [f for f in self.features_include if f not in excl]


def _month_diff(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def load_config(path):
    """Read + validate a YAML config. Fails loud before any Spark read or fit."""
    with open(path) as fh:
        return RunConfig(**yaml.safe_load(fh))
