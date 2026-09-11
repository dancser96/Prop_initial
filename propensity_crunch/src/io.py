"""Spark read boundary. Prune to (id + feature_cols [+ extra cols]) and the
eligible population IN SPARK, then collect. Never .toPandas() the full width.

feature_cols lets callers choose the projection: exploration passes
cfg.candidates (pre-decision), modelling defaults to cfg.features (survivors).
Eligibility is applied on every read, so obs/OOT/inference stay consistent."""
from __future__ import annotations

import pandas as pd
from pyspark.sql import functions as F


def _read(spark, ref):
    return spark.read.parquet(ref) if "/" in ref else spark.table(ref)


def load_snapshot(spark, cfg, month, extra_cols=None, feature_cols=None) -> pd.DataFrame:
    feats = cfg.features if feature_cols is None else list(feature_cols)
    df = _read(spark, cfg.table).filter(F.col(cfg.month_col) == str(month))
    if cfg.eligibility_expr:
        df = df.filter(cfg.eligibility_expr)

    cols = [cfg.id_col] + feats + list(extra_cols or [])
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"config names columns absent from {cfg.table}: {sorted(missing)}")
    return df.select(*cols).toPandas()
