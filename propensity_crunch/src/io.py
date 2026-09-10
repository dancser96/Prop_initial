"""Spark read boundary. Prune to (id + candidate features) and eligible
population IN SPARK, then collect. Never .toPandas() the full width — that is
the one read that OOMs a single-node crunch."""
from __future__ import annotations

import pandas as pd
from pyspark.sql import functions as F


def load_snapshot(spark, cfg, month) -> pd.DataFrame:
    df = spark.table(cfg.table).filter(F.col(cfg.month_col) == str(month))
    if cfg.eligibility_expr:
        df = df.filter(cfg.eligibility_expr)          # eligible population only

    cols = [cfg.id_col] + cfg.features
    missing = set(cols) - set(df.columns)
    if missing:                                        # config resolves against real table
        raise ValueError(
            f"config names columns absent from {cfg.table}: {sorted(missing)}"
        )
    return df.select(*cols).toPandas()
