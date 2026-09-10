# Propensity crunch scaffold

Crunch-mode propensity scoring for ~10 retail products, built so the reusable
seams transfer to the real framework later. **Not** the framework. Deliberately thin.

## Stages
Three stage notebooks (one each, product per section) + one modelling notebook per product:
- **Data Creation** — decoupled, single-in / single-out Spark job that materialises
  target (and engineered feature) columns via month rollups. Separate from the spine
  on purpose; with a feature store this logic would live there.
- **EDA** — approachable visual profiling via reusable helpers.
- **Feature Checks** — first-order leakage/quality screen; the human-decided exclude list.
- **Modelling** — one notebook per product; a thin caller of `run(cfg, spark)`.

## The two things built "properly" (these transfer verbatim)
- `configs/_schema.py` — ONE validated config (the framework's config spine).
- `src/run.py`         — ONE `run(cfg, spark)` entrypoint (the framework's run-harness).

## Layout
```
configs/
  _schema.py          # Pydantic v2 schema + validator  [SEAM 1]
  fx_activation.yaml  # template config; copy per product
  credit_cards.yaml
src/
  rollups.py          # Spark forward/backward month rollups + binary target (Data Creation)
  io.py               # Spark prune-then-collect (table OR path); OOM guard
  dataset.py          # load_labelled(): reads features + materialised target
  contract.py         # runtime tripwires (fail loud, early)
  leakage.py          # first-order flag report (human decides, never auto-drop)
  plots.py            # small reusable matplotlib EDA helpers
  evaluate.py         # external OOT eval vs population-base-rate baseline
  run.py              # orchestrator                     [SEAM 2]
  card.py             # minimal model card + plots
notebooks/
  Data_Creation.ipynb
  EDA.ipynb
  Feature_Checks.ipynb
  Modelling_FX_Activation.ipynb
run_product.py        # CLI equivalent: python run_product.py configs/fx_activation.yaml
artifacts/<product>/<run_id>/   # frozen config, leakage report, scores, model, card
```

## Data flow
Data Creation writes one snapshot per product to an HDFS path → the product YAML's
`table` points at it → EDA / Feature Checks / Modelling read it via config. The target
is a column in that snapshot; the spine never rebuilds it.

## What you fill before a first run
1. Data Creation: real `INPUT_PATH`, the activity columns, windows, and thresholds.
2. Each product YAML: real `table`, `features_include`, `eligibility_expr`, `target_col`.

## Invariants baked in
- Config validated on load; bad temporal order / sub-window OOT gap / missing column fails fast.
- Target is strictly forward (features as-of obs month, label over the window after); panel-edge
  rows without a full window are nulled and dropped.
- OOT on a later month; metric + baseline declared in config before results.
- Leakage screen flags, never auto-drops. No automated feature selection.
- Score log (score + config_hash + feature signature) written at inference.

## Assumptions to verify
- Pydantic **v2** (validator decorators differ on v1).
- FLAML feature-importance extraction is engine-dependent → best-effort in `card.py`.
- `io.py` treats a `table` containing "/" as a parquet path, else a Hive table name.
