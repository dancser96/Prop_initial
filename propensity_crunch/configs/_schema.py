"""SEAM 1 — the config spine.

ONE schema all 10 product configs conform to, validated on load. This is the
piece that transfers verbatim to the framework. Pydantic v2.

What it validates: config SHAPE + temporal-safety preconditions. NOT the data.
(Column-resolution against the real snapshot is a runtime check in io.py, since
it needs the table.)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_METRICS = {"roc_auc", "average_precision", "log_loss"}


class TargetSpec(BaseModel):
    # ONE parameterised definition. Divergent EVENT semantics (loan origination
    # is an event, not a usage threshold) get a second NAMED body in target.py,
    # dispatched on `kind` — never an inline if-product branch in a caller.
    kind: Literal["usage_threshold", "product_flag", "origination"]
    window_months: int = Field(ge=1, description="activation window after obs_date")
    min_usage: Optional[float] = None       # kind=usage_threshold
    flag_col: Optional[str] = None          # kind=product_flag / origination
    activity_table: Optional[str] = None    # forward-window source, if needed


class ModelSpec(BaseModel):
    time_budget: int = Field(ge=30, description="FLAML seconds")
    estimator_list: list[str] = ["lgbm", "xgboost", "rf", "extra_tree"]
    n_splits: int = 5
    seed: int = 42


class RunConfig(BaseModel):
    product: str
    table: str                      # Hive feature snapshot (parquet-backed)
    id_col: str = "cif"
    month_col: str = "snapshot_month"

    obs_date: date                  # train / within-month CV
    oot_date: date                  # OOT validation month
    infer_date: date                # scoring month

    features_include: list[str]     # curated candidate set — NO auto-selection
    features_exclude: list[str] = []
    eligibility_expr: Optional[str] = None   # product-specific Spark filter, hardcoded

    target: TargetSpec
    model: ModelSpec

    metric: str = "roc_auc"                          # declared BEFORE results
    baseline: Literal["population_base_rate"] = "population_base_rate"
    min_base_rate: float = 0.001                     # sanity-tripwire band
    max_base_rate: float = 0.60

    @field_validator("features_include")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("features_include is empty")
        return v

    @field_validator("metric")
    @classmethod
    def _metric_allowed(cls, v):
        if v not in ALLOWED_METRICS:
            raise ValueError(f"metric {v!r} not in {sorted(ALLOWED_METRICS)}")
        return v

    @model_validator(mode="after")
    def _temporal_safety(self):
        if not (self.obs_date < self.oot_date < self.infer_date):
            raise ValueError("require obs_date < oot_date < infer_date")
        gap = _month_diff(self.obs_date, self.oot_date)
        if gap < self.target.window_months:
            raise ValueError(
                f"oot_date must be >= window_months ({self.target.window_months}m) "
                f"after obs_date so train/OOT activation windows do not overlap; "
                f"got {gap}m"
            )
        return self

    @property
    def features(self) -> list[str]:
        excl = set(self.features_exclude)
        return [f for f in self.features_include if f not in excl]


def _month_diff(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def load_config(path: str | Path) -> RunConfig:
    """Fails loud on a bad config — before any Spark read or FLAML run."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return RunConfig(**raw)
