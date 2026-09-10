"""The ~10-line per-product entrypoint — replaces logic-in-notebook.

    python run_product.py configs/credit_cards.yaml

Each product notebook, if you keep one, is just this call plus whatever ad-hoc
plots you want to eyeball. Logic lives in src/, never in the notebook.
"""
import sys

from pyspark.sql import SparkSession

from configs._schema import load_config
from src.run import run

if __name__ == "__main__":
    cfg = load_config(sys.argv[1])            # validates on load; fails loud on bad config
    spark = SparkSession.builder.getOrCreate()
    result = run(cfg, spark)
    print(result["run_id"], "->", result["out"])
    print(result["metrics"])
