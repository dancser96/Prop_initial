"""SEAM 1 — the config spine. ONE schema all product configs conform to,
validated on load. Pydantic v2.

Target creation is decoupled from this spine: the target is materialised as a
column during the Data Creation stage, and the config only names that column
(`target_col`) plus the forward window length (`window_months`) it was built
with. `window_months` is still needed here to enforce the train/OOT
no-overlap rule.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_METRICS = {"roc_auc", "average_precision", "log_loss"}


class ModelSpec(BaseModel):
    time_budget: int = Field(ge=30, description="FLAML seconds")
    estimator_list: list[str] = ["lgbm", "xgboost", "rf", "extra_tree"]
    n_splits: int = 5
    seed: int = 42


class RunConfig(BaseModel):
    product: str
    table: str                      # Hive table name OR HDFS parquet path of the snapshot
    id_col: str = "cif"
    month_col: str = "snapshot_month"

    obs_date: date                  # train / within-month CV
    oot_date: date                  # OOT validation month
    infer_date: date                # scoring month

    features_include: list[str]     # curated candidate set — NO automated selection
    features_exclude: list[str] = []
    eligibility_expr: Optional[str] = None   # product-specific Spark filter, hardcoded

    target_col: str                 # column materialised during Data Creation
    window_months: int = Field(ge=1)  # forward window the target was built with

    model: ModelSpec

    metric: str = "roc_auc"                     # declared BEFORE results
    baseline: str = "population_base_rate"
    min_base_rate: float = 0.001                # sanity-tripwire band
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
        if gap < self.window_months:
            raise ValueError(
                f"oot_date must be >= window_months ({self.window_months}m) after "
                f"obs_date so train/OOT target windows do not overlap; got {gap}m"
            )
        return self

    @property
    def features(self) -> list[str]:
        excl = set(self.features_exclude)
        return [f for f in self.features_include if f not in excl]


def _month_diff(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def load_config(path):
    """Fails loud on a bad config — before any Spark read or FLAML run."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return RunConfig(**raw)
