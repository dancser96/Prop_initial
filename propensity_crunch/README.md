# Propensity crunch scaffold

Crunch-mode propensity scoring for ~10 retail products, built so the reusable
seams transfer to the real framework later. **Not** the framework. Deliberately thin.

See **DOCS.md** for the detailed design, config reference, and rationale.

## Stages
- **Data Creation** — one input panel → target/feature columns via month rollups → one
  enriched panel out, covering all products. Decoupled from the spine.
- **EDA** — approachable visual profiling via reusable helpers.
- **Feature Checks** — first-order leakage/quality screen; the human-decided exclude list.
- **Modelling** — one notebook per product; a thin caller of `run(cfg, spark)`.
- **Inference** — batch-scores eligible clients across a configurable product list at an
  inference date; writes long format `(cif, product, model_hash, infer_date, score)`.
  Decoupled from training (reuses frozen config + saved model); no retraining.

## The two things built "properly" (transfer verbatim)
- `configs/_schema.py` — ONE validated config (the framework's config spine).
- `src/run.py`         — ONE `run(cfg, spark)` entrypoint (the framework's run-harness).

## Layout
```
configs/
  _schema.py          # Pydantic v2 schema + validator  [SEAM 1]
  _TEMPLATE.yaml      # annotated blank config — copy per product
  fx_activation.yaml  # worked example (no downsampling)
  credit_cards.yaml   # worked example (imbalanced target + downsample)
src/
  rollups.py          # Spark forward/backward rollups + binary target (Data Creation)
  io.py               # prune-then-collect read (table OR path); applies eligibility
  dataset.py          # load_labelled + train-only downsample
  contract.py         # runtime tripwires (fail loud, early)
  leakage.py          # first-order flag report (human decides, never auto-drop)
  plots.py            # reusable matplotlib EDA helpers
  evaluate.py         # external OOT eval vs population-base-rate baseline
  run.py              # orchestrator                     [SEAM 2]
  card.py             # model card + plots
  inference.py        # batch scoring: resolve model, score eligible clients, long format
notebooks/
  Data_Creation.ipynb  EDA.ipynb  Feature_Checks.ipynb  Modelling_FX_Activation.ipynb
  Inference.ipynb
run_product.py        # CLI: python run_product.py configs/fx_activation.yaml
artifacts/<product>/<run_id>/   # frozen config, leakage report, scores, model, card, search log
```

## Data flow
Data Creation writes ONE enriched panel → every product YAML's `table` points at it →
EDA / Feature Checks / Modelling read their slice via config. The target is a column in
that panel; the spine never rebuilds it.

## What you fill before a first run
1. Data Creation: real `INPUT_PATH`, activity columns, windows, thresholds.
2. Each product YAML: real `table`, `features_include`, `eligibility_expr`, `target_col`
   (start from `configs/_TEMPLATE.yaml`).

## Invariants
Config validated on load · strictly-forward target with panel-edge guard · OOT on a later
month · metric + baseline declared before results · leakage flags never auto-drop · no
automated feature selection · score log written at inference.

## Assumptions
Pydantic v2 · FLAML importance is best-effort · `table` with "/" = parquet path, else Hive.
