"""Small, reusable matplotlib EDA helpers (no seaborn dependency).

Each returns the Axes so notebook sections look identical across products.
Do not set a backend here — let the notebook's inline backend display them.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_missingness(df, cols=None, top=20, ax=None):
    cols = cols or list(df.columns)
    miss = df[cols].isna().mean().sort_values(ascending=False).head(top)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2, 0.3 * len(miss))))
    miss.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("fraction missing")
    ax.set_title("Missingness (top features)")
    return ax


def plot_hist(x, by=None, bins=30, ax=None):
    """Histogram of a numeric feature; if `by` (a 0/1 label) is given, overlay
    one histogram per class to see whether the groups separate."""
    x = pd.Series(x)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if by is None:
        ax.hist(x.dropna(), bins=bins)
    else:
        by = pd.Series(by).reset_index(drop=True)
        xv = x.reset_index(drop=True)
        for v in sorted(by.dropna().unique()):
            ax.hist(xv[by == v].dropna(), bins=bins, alpha=0.5, label=f"target={v}")
        ax.legend()
    ax.set_title(f"Distribution — {x.name}")
    ax.set_xlabel(x.name)
    return ax


def plot_target_rate(x, y, bins=10, ax=None):
    """Bin a numeric feature and plot the target rate per bin. The dashed line
    is the overall base rate; a monotonic trend suggests real signal."""
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    try:
        b = pd.qcut(x, bins, duplicates="drop")
    except Exception:
        b = pd.cut(x, bins)
    rate = y.groupby(b, observed=True).mean()
    rate.plot.bar(ax=ax)
    ax.axhline(float(y.mean()), ls="--", lw=1, color="k")
    ax.set_ylabel("target rate")
    ax.set_title(f"{x.name} vs target rate")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    return ax


def plot_corr(df, cols=None, ax=None):
    cols = cols or df.select_dtypes("number").columns.tolist()
    corr = df[cols].corr()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation (numeric features)")
    return ax
