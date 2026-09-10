"""Spark helpers for the Data Creation stage: forward/backward month rollups
and a binary-target threshold.

All operate on a single monthly-panel dataframe (one row per id per month) via
window functions — single dataframe in, single dataframe out, no joins to other
tables.

Anchoring: `add_month_index` turns the month column into a contiguous integer
so a window `rangeBetween` counts in whole months.
  - `forward_rollup`  aggregates the NEXT `months` months  -> targets
  - `backward_rollup` aggregates the PREVIOUS `months` months -> features
Missing months in the panel are simply absent from the range; the window stays
correct.
"""
from __future__ import annotations

from pyspark.sql import Window, functions as F

_AGGS = {
    "sum": F.sum, "count": F.count, "avg": F.mean, "mean": F.mean,
    "max": F.max, "min": F.min,
}


def add_month_index(df, month_col="snapshot_month", out_col="month_idx"):
    d = F.to_date(F.col(month_col))
    return df.withColumn(out_col, (F.year(d) * F.lit(12) + F.month(d)).cast("int"))


def forward_rollup(df, id_col, value_col, months, out_col,
                   order_col="month_idx", agg="sum"):
    w = Window.partitionBy(id_col).orderBy(order_col).rangeBetween(1, months)
    return df.withColumn(out_col, _AGGS[agg](F.col(value_col)).over(w))


def backward_rollup(df, id_col, value_col, months, out_col,
                    order_col="month_idx", agg="sum"):
    w = Window.partitionBy(id_col).orderBy(order_col).rangeBetween(-months, -1)
    return df.withColumn(out_col, _AGGS[agg](F.col(value_col)).over(w))


def binary_target(df, signal_col, threshold, out_col):
    return df.withColumn(out_col, (F.col(signal_col) >= F.lit(threshold)).cast("int"))
