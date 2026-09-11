# Propensity Crunch — Design & Documentation

Initial documentation for the crunch-mode propensity scaffold. README is the
quick map; this is the "why and how" one level deeper.

---

## 1. What this is (and is not)

A deliberately thin setup to score ~10 retail products fast, while banking the
seams that carry over to the real framework. It is **not** the framework: no
component registry, no orchestration, no feature store, no LLM branch. Two
things are built properly because they transfer verbatim; everything else is
disposable and hardcoded per product where needed.

- `configs/_schema.py` — the **config spine** (validated config).
- `src/run.py` — the **run-harness** (`run(cfg, spark)`).

Guiding rule: **logic lives in `src/`, notebooks only call and inspect.** The
import boundary enforces "no functions defined in notebook sections."

---

## 2. Architecture

**Data Creation (decoupled, raw Spark).** One monthly panel in → add target and
engineered-feature columns via month rollups → one enriched panel out, covering
all products. Separate from the spine on purpose; a feature store would own this.

**The spine (config-driven).** Per product: read its slice of the enriched panel
→ labelled frame → correctness checks → FLAML → external OOT eval + importance →
inference + provenance. The spine never rebuilds the target; it reads the column
Data Creation materialised.

```
raw monthly panel ──[Data Creation + src/rollups.py]──▶ enriched panel (ONE table)
                                                              │
   per-product config (table, month, features, eligibility, target_col)
                                                              │
   EDA ─ Feature Checks ─ Modelling(run) ──▶ artifacts/<product>/<run_id>/
```

---

## 3. Module reference

| File | Role |
|---|---|
| `configs/_schema.py` | Config schema + validation (SEAM 1). |
| `src/rollups.py` | Spark forward/backward month rollups + binary target (Data Creation). |
| `src/io.py` | Prune-then-collect read; applies eligibility; picks the feature projection. |
| `src/dataset.py` | `load_labelled` (features + target); `downsample_train` (train-only). |
| `src/contract.py` | Runtime tripwires (unique id, non-constant target, base-rate band). |
| `src/leakage.py` | First-order leakage/quality screen (flags, never drops). |
| `src/feature_stats.py` | VIF, mutual information, univariate summary, event-rate-by-category, signal persistence (obs→OOT). |
| `src/plots.py` | Reusable matplotlib EDA helpers. |
| `src/calibration.py` | Score calibration behind a swappable registry (`prior_shift`). |
| `src/evaluate.py` | OOT metrics: AUC + top-1% / top-decile precision/recall/lift. |
| `src/importance.py` | Model-agnostic permutation importance (OOT). |
| `src/run.py` | Orchestrator (SEAM 2). |
| `src/card.py` | Model card + importance plot. |
| `src/inference.py` | Batch scoring: resolve model, score eligible clients, long format. |

---

## 4. Two feature projections of one config

The single most important config detail. A config carries the full candidate
list and the human-decided exclusions; code reads **two views** of it:

- `cfg.candidates` = `features_include` — the pre-decision candidate set.
  **EDA and Feature Checks read this**, so a feature stays visible even after you
  exclude it (you can re-check the decision that dropped it).
- `cfg.features` = `features_include − features_exclude` — the survivors.
  **`run()` reads this** — it is the actual model input.

The loop: seed `features_include` generously once; screen candidates in EDA /
Feature Checks; write drops into `features_exclude` (and new ideas into
`features_include`); modelling reads the survivors. One file, one source of
truth. `run()` freezes the resolved config per run, so editing the file never
rewrites history.

---

## 5. Config reference — how to produce one

Copy `configs/_TEMPLATE.yaml` to `<product>.yaml`, fill the placeholders;
`load_config` validates on load. Fields:

| Field | Meaning |
|---|---|
| `product` | Short id; names the artifacts folder. |
| `table` | Hive table name **or** HDFS path of the enriched panel (`/` → path). |
| `id_col` | Customer key (e.g. `cif`). |
| `month_col` | Column stamping each row's **monthly snapshot** (one row per id per month). Selects obs/OOT/inference months; anchors the rollups. |
| `obs_date` / `oot_date` / `infer_date` | Train / OOT / scoring months. Validated: `obs < oot < infer` and `(oot − obs) ≥ window_months`. |
| `features_include` / `features_exclude` | Candidate set and human-decided drops (see §4). |
| `eligibility_expr` | Spark filter for the modellable population (§7). |
| `target_col` / `window_months` | Materialised target column and its forward window. |
| `model` | FLAML settings (§6). |
| `downsample` | Optional, train-only (§7). |
| `calibration` | Method applied automatically when `downsample` is set (§7). |
| `metric` | Optimisation + OOT metric, declared before results. |
| `min/max_base_rate` | Sanity band for the target base rate. |

Examples: `fx_activation.yaml` (no downsampling), `credit_cards.yaml`
(imbalanced target → downsample + calibration).

---

## 6. Model params — what we set and what we don't

`model:` exposes the knobs that matter: `time_budget` (main lever),
`estimator_list`, `eval_method` + `n_splits` (cv is clean because one training
month = one row per CIF), `ensemble`, `sample` (FLAML search-time subsampling,
not class balancing), `early_stop`, `n_jobs`, `seed`. Left at FLAML defaults on
purpose: custom hyperparameter spaces, starting points, per-estimator overrides,
custom splitters. `metric` sits at the top level because it is governance —
declared before results and reused for OOT. The valid optimisation metrics
(`roc_auc`, `ap`, `log_loss`, `f1`, `macro_f1`, `micro_f1`, `accuracy`) are
listed inline in `configs/_TEMPLATE.yaml`. The FLAML search log is written to
`artifacts/.../flaml_search.log`, and the card records `model_selected`
(`best_estimator`, `best_config` = winning hyperparameters, `best_cv_loss`) so
the artifact is self-describing without unpickling.

---

## 7. Eligibility, downsampling, calibration

**Eligibility — config, applied on every read.** `eligibility_expr` is a Spark
filter applied in `io.py` for obs, OOT, and inference alike, so the population
definition is consistent by construction. Keep it to plain column conditions.

**Downsampling — config, train-only.** `downsample.neg_per_pos` keeps all
positives and samples negatives to that ratio, in `run.py`, on the **training
frame only** — OOT and inference stay on the full eligible population, so
evaluation and scored volumes are honest. Use it for very imbalanced targets.

**Calibration — automatic, paired with downsampling.** Downsampling inflates the
prior odds, so predicted probabilities would be too high. When `downsample` is
set, `run()` applies `cfg.calibration` (default `prior_shift`) automatically: a
log-odds offset of `logit(true_rate) − logit(sampled_rate)` maps scores back to
the true base rate. Exact for the sampling shift, no held-out set needed, and
rank-based OOT metrics are unaffected. The score log carries both `score`
(calibrated) and `score_raw`. Calibration is behind a registry
(`src/calibration.py`) so isotonic/Platt can be added later without touching
callers. Without downsampling there is no prior shift, so no calibration is
applied.

Never downsample in Data Creation (biases the shared panel) or on
OOT/inference.

---

## 8. Metrics

`oot_evaluate` reports, on the OOT month:

- **AUC** — overall ranking quality.
- **top-1% and top-decile** — `precision`, `recall`, and `lift` at the top of
  the score ranking. Given the retail base size, these are the operational
  numbers: precision = hit rate among the slice you'd contact, recall = share of
  all activators captured in that slice, lift = precision ÷ base rate. Baseline
  is the population base rate (lift reference 1.0).

All are rank-based, so calibration does not change them.

---

## 9. The stage notebooks

- **Data Creation** — global read + month index at top, per-product sections
  that only *add* columns, single global write at bottom. Produces the one
  enriched panel all configs point to.
- **EDA** — reads `cfg.candidates`; profiles with `src/plots.py`; adds a compact
  univariate summary (nulls + quantiles), event-rate-by-category tables, **VIF**
  (multicollinearity beyond pairwise correlation) and **mutual information**
  (non-linear target dependence). Makes no changes — output is decisions.
- **Feature Checks** — reads `cfg.candidates`; runs the leakage/quality screen
  plus a **signal-persistence check** (univariate AUC train-vs-OOT + PSI drift),
  which catches features that are strong in-sample but decay or drift by the OOT
  month. A human writes the `features_exclude` list back into the config. Flags,
  never drops; no automated selection.
- **Modelling_<product>** — thin caller of `run(cfg, spark)`; displays OOT
  metrics, model-agnostic permutation importance, leakage report, card, scores.
- **Inference** — batch scoring, decoupled from training. Loops a configurable
  product list, scores eligible clients at an inference date using each product's
  **frozen config + saved model + calibration offset**, and writes one
  long-format table `(cif, product, model_hash, infer_date, score)` partitioned
  by product. `model_hash` is the content hash of that run's `model.pkl`, pinning
  the exact model behind every score. Reproducible: later YAML edits never change
  what a given `model_hash` produces. Re-score a new month by overriding the
  inference date; the same trained models score the fresh snapshot.

---

## 10. Governance invariants

1. Config validated on load (temporal order, OOT gap, missing columns fail fast).
2. Target is strictly forward; panel-edge rows without a full window are nulled/dropped.
3. OOT evaluation on a later month, baseline + metric declared before results.
4. Correctness wraps the engine: leakage screen, permutation importance, and OOT
   are external; FLAML sees only curated features and never grades its own OOT.
5. Leakage screen flags for review; never auto-drops; no automated selection.
6. Provenance: frozen config, config hash, feature-set signature, score log
   (`score` + `score_raw`) written per run.

---

## 11. Assumptions & caveats

- Pydantic **v2** (validator decorators differ on v1).
- Data Creation assumes engineered features are columns in the single panel via
  rollups; features from separate activity tables need an upstream join.
- `io.py` treats a `table` value containing `/` as a parquet path, else Hive.
- Single-month training bakes in that month's seasonality — note it, don't
  over-trust the level.
- Permutation importance subsamples OOT (`max_rows`) for speed.
- Rollup convention: backward windows **include** the current month (trailing N);
  forward windows **exclude** it (strictly future). Backward features at the
  panel's earliest months rest on partial windows — keep obs/OOT clear of the
  panel start.
- Eligibility × features: a backward feature built on the same activity that
  defines eligibility is constant for the eligible population (e.g. FX-activity
  features under `is_fx_active = 0`). Check EDA/VIF flags it, and prefer features
  the eligible population can actually vary on.
