# Propensity crunch scaffold

Crunch-mode propensity scoring for ~10 retail products, built so the reusable
seams transfer to the real framework later. **Not** the framework. Deliberately thin.

## The only two things built "properly" (these transfer verbatim)
- `configs/_schema.py`  — ONE validated config schema (the framework's config spine).
- `src/run.py`          — ONE `run(cfg, spark)` entrypoint (the framework's run-harness).

Everything else is disposable and hardcoded per-product where needed.

## Layout
```
configs/
  _schema.py          # Pydantic v2 schema + validator  [SEAM 1]
  credit_cards.yaml   # copy this per product -> 10 files
src/
  io.py               # Spark prune-then-collect (OOM guard)
  contract.py         # runtime tripwires (fail loud, early)
  target.py           # STUB — target bodies owned by DS, not framework author
  leakage.py          # first-order flag report (human decides, never auto-drop)
  evaluate.py         # external OOT eval vs population-base-rate baseline
  run.py              # orchestrator                     [SEAM 2]
  card.py             # minimal model card + plots
run_product.py        # thin caller: python run_product.py configs/credit_cards.yaml
artifacts/<product>/<run_id>/   # frozen config, leakage report, scores, model, card
```

## What you must fill before a first run
1. `src/target.py` bodies (the 3 `NotImplementedError`s) — semantics, not framework.
2. Real `table`, `features_include`, `eligibility_expr` in each product YAML.

## Invariants baked in (the point of the exercise)
- Config validated on load; bad temporal order / dropped feature fails in seconds.
- Features as-of `obs_date`, label strictly after (contract + target window).
- OOT on a later month; metric + baseline declared in config *before* results.
- Leakage screen flags, never auto-drops. No automated feature selection.
- Score log (score + config_hash + feature signature) written at inference.

## Assumptions to verify
- Pydantic **v2** (validator decorators differ on v1).
- FLAML feature-importance extraction is engine-dependent → best-effort in `card.py`.
