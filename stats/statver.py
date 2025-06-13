"""
ga_stats_table.py

Usage
    python ga_stats_table.py experiment.csv
    python ga_stats_table.py experiment.xlsx

The input file must have three columns named
    Generation   Individual   Fitness

What the script produces
  • One pandas DataFrame called summary that holds per generation data
  • A run_stats dictionary that describes the whole run
  • The DataFrame is also written to <original name>_per_gen.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Load the source file whether it is CSV or Excel."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    expected = {"Generation", "Individual", "Fitness"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Input file must contain columns {expected}")
    return df


def per_generation(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("Generation")["Fitness"]
    summary = group.agg(
        pop_size="count",
        best="max",          # highest value is best
        worst="min",
        mean="mean",
        median="median",
        stdev="std",
    ).sort_index()

    # step-by-step improvement
    diff = summary["best"].diff()
    pct  = diff / summary["best"].shift(1).abs() * 100
    summary["best_improvement"]      = diff.fillna(0)
    summary["best_improvement_pct"]  = pct.fillna(0)

    # cumulative improvement from generation 0  <<< add these two lines
    baseline = summary["best"].iloc[0]
    summary["best_cumulative_pct"] = (summary["best"] - baseline) / abs(baseline) * 100

    return summary



def run_stats(summary: pd.DataFrame) -> dict:
    """Compute run level numbers."""
    best_val = summary["best"].max()
    best_gen = int(summary["best"].idxmax())
    final_mean = summary["mean"].iloc[-1]

    first_best = summary["best"].iloc[0]
    final_best = summary["best"].iloc[-1]
    total_best_gain = final_best - first_best
    total_best_pct = total_best_gain / abs(first_best) * 100

    return {
        "best_fitness": best_val,
        "generation_of_best": best_gen,
        "final_generation_mean": final_mean,
        "total_best_improvement": total_best_gain,
        "total_best_pct_improvement": total_best_pct,
        "number_of_generations": len(summary),
        "final_best_cumulative_pct": total_best_pct, 
    }


def main(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    raw = read_table(path)
    summary = per_generation(raw)
    stats = run_stats(summary)

    out_csv = path.with_stem(path.stem + "_per_gen").with_suffix(".csv")
    summary.to_csv(out_csv, index=True)

    print("Per generation statistics written to", out_csv)
    print("\nRun summary\n")
    for k, v in stats.items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide the csv or xlsx file")
        sys.exit(1)
    main(sys.argv[1])