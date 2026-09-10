"""Spark read boundary. Prune to (id + candidate features [+ extra cols]) and
the eligible population IN SPARK, then collect. Never .toPandas() the full
width — that is the read that OOMs a single-node crunch."""
from __future__ import annotations

import pandas as pd
from pyspark.sql import functions as F


def _read(spark, ref):
    # Hive table name (schema.table) or an HDFS/parquet path.
    return spark.read.parquet(ref) if "/" in ref else spark.table(ref)


def load_snapshot(spark, cfg, month, extra_cols=None) -> pd.DataFrame:
    df = _read(spark, cfg.table).filter(F.col(cfg.month_col) == str(month))
    if cfg.eligibility_expr:
        df = df.filter(cfg.eligibility_expr)          # eligible population only

    cols = [cfg.id_col] + cfg.features + list(extra_cols or [])
    missing = set(cols) - set(df.columns)
    if missing:                                        # config resolves against real snapshot
        raise ValueError(
            f"config names columns absent from {cfg.table}: {sorted(missing)}"
        )
    return df.select(*cols).toPandas()
