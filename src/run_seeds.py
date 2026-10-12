"""Runs experiments over 5 seeds, logs with MLflow, and exports results_summary.csv."""
import os, sys, subprocess
import mlflow, pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.train import main

SEEDS = [0, 1, 2, 3, 42]
DB_PATH = "sqlite:///" + os.path.abspath("mlflow.db")

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    commit = "unknown"

mlflow.set_tracking_uri(DB_PATH)
mlflow.set_experiment("session5-repro-lab")

for seed in SEEDS:
    with mlflow.start_run(run_name=f"seed-{seed}"):
        r = main(seed)
        mlflow.log_params({"model": "logistic_regression", "seed": seed, "git_commit": commit, "data_md5": r["data_md5"]})
        mlflow.log_metrics({"accuracy_mean": r["accuracy_mean"], "accuracy_sd": r["accuracy_sd"]})

runs = mlflow.search_runs(experiment_names=["session5-repro-lab"])
summary = runs[["params.seed", "metrics.accuracy_mean", "metrics.accuracy_sd", "params.git_commit"]].sort_values("params.seed")
summary.to_csv("results_summary.csv", index=False)

acc = runs["metrics.accuracy_mean"]
print("✅ Table generated successfully:")
print(summary.to_string(index=False))
print(f"\nHeadline Metric (Across 5 Seeds): {acc.mean():.4f} ± {acc.std(ddof=1):.4f}")
