"""STUB — owned by the DS building target logic, NOT the framework author.

ONE signature, parameterised by TargetSpec. Divergent EVENT semantics get a
second NAMED body dispatched on `kind` below — never an inline if-product
branch in a caller.

Contract every body MUST satisfy (enforced downstream by contract.py):
  - returns a pd.Series of 0/1 int, INDEXED BY id_col
  - the activation window is measured STRICTLY AFTER obs_date
    (features are as-of obs_date; the label comes from the forward window only)
  - a CIF with no forward activation event => 0 (handled in run.py via fillna)
"""
from __future__ import annotations

import pandas as pd


def build_target(spark, cfg, month) -> pd.Series:
    spec = cfg.target
    if spec.kind == "usage_threshold":
        return _usage_threshold(spark, cfg, month)
    if spec.kind == "product_flag":
        return _product_flag(spark, cfg, month)
    if spec.kind == "origination":
        return _origination(spark, cfg, month)
    raise ValueError(f"unknown target kind {spec.kind!r}")


def _usage_threshold(spark, cfg, month) -> pd.Series:
    # label=1 if forward activity in (month, month+window] >= spec.min_usage
    raise NotImplementedError("usage_threshold target: fill in")


def _product_flag(spark, cfg, month) -> pd.Series:
    # label=1 if spec.flag_col turns on in (month, month+window]
    raise NotImplementedError("product_flag target: fill in")


def _origination(spark, cfg, month) -> pd.Series:
    # label=1 if first origination event falls in (month, month+window]
    raise NotImplementedError("origination target: fill in")
