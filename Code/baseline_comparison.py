"""
Nomura Quant Challenge 5 - Market Making
Baseline comparison harness for Task 5
=======================================

Purpose
-------
Reproducibly test THREE quoting strategies head-to-head and produce the exact
numbers quoted on the CV (objective beats fill-model by ~8% PnL and static by
~19% PnL on the regime-shift stream; stationary-grid mean Sharpe-like scores).

The three strategies (all implemented inside task5.quote via its `mode` switch):

  * static     : quote a FIXED multiple of volatility, X_STATIC * sigma. No
                 learning at all. The "do nothing clever" baseline.
  * fillmodel  : estimate the hidden fill-decay gamma ONLINE and quote at the
                 half-spread x* = 1/gamma. This maximises the CAPTURED SPREAD
                 proxy x*exp(-gamma*x) only -- it ignores realised mid-edge.
  * objective  : my strategy. Same online gamma, but target the objective
                 optimum x* = 1/gamma - beta(alpha): an adversity-driven edge
                 correction that quotes tighter on benign flow and wider on
                 toxic flow, plus an Avellaneda-Stoikov inventory skew. This
                 maximises realised PnL, which is what the challenge scores.

Two studies (identical to task5.validate_quote, isolated here for auditing):
  A. stationary (lambda, gamma, phi) grid  -> regression check
  B. three-regime discontinuous-shift stream -> robustness check

Outputs
-------
  baseline_comparison_results.csv   per-configuration metrics
  prints the headline percentage gaps used on the CV

Run:
  TRADE_DATA_PATH=../trade_data.csv python baseline_comparison.py
"""

import os
import numpy as np
import pandas as pd

import task5

# Same chronological data stream task5 uses (sigma, alpha, eta, mid_close).
_DATA_PATH = os.environ.get(
    "TRADE_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_data.csv"),
)

# Stationary (lambda, gamma, phi) grid -- the regression check.
STATIONARY_GRID = [
    (0.5, 0.7, 1e-6),
    (0.5, 1.0, 1e-6),
    (0.6, 1.5, 1e-6),
    (0.5, 2.0, 1e-6),
    (0.7, 0.8, 1e-5),
]

# Three discontinuous regimes: easy -> hard+toxic+high-phi -> medium.
REGIME_SCHEDULE = [
    {"frac": 0.34, "lam": 0.80, "gam": 0.60, "phi": 1e-6, "tox": 1.00},
    {"frac": 0.33, "lam": 0.35, "gam": 2.80, "phi": 5e-4, "tox": 0.40},
    {"frac": 0.33, "lam": 0.60, "gam": 1.30, "phi": 5e-6, "tox": 1.00},
]

VARIANTS = ["static", "fillmodel", "objective"]


def run_study_A(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary grid: each strategy on each fixed (lambda, gamma, phi)."""
    rows = []
    for (lam, gam, phi) in STATIONARY_GRID:
        for mode in VARIANTS:
            r = task5.backtest(df, lam, gam, phi, seed=7, mode=mode)
            rows.append({
                "study": "A-stationary", "variant": mode,
                "lam": lam, "gam": gam, "phi": phi,
                "total_pnl": round(r["total_pnl"], 1),
                "score": round(r["score"], 2),
                "max_dd": round(r["max_drawdown"], 1),
            })
    return pd.DataFrame(rows)


def run_study_B(df: pd.DataFrame):
    """Regime-shift stream: each strategy on the same discontinuous schedule."""
    reg = task5._regime_arrays(df, REGIME_SCHEDULE)
    rows, results = [], {}
    for mode in VARIANTS:
        r = task5._run(df, reg["lam"], reg["gam"], reg["phi"], reg["tox"],
                       seed=7, mode=mode, spsa=False,
                       reg_idx=reg["idx"], n_reg=len(REGIME_SCHEDULE))
        results[mode] = r
        rows.append({
            "study": "B-regime-shift", "variant": mode,
            "lam": "shift", "gam": "shift", "phi": "shift",
            "total_pnl": round(r["total_pnl"], 1),
            "score": round(r["score"], 2),
            "max_dd": round(r["max_drawdown"], 1),
        })
    return pd.DataFrame(rows), results


def main():
    df = task5._prepare_stream(_DATA_PATH)

    a = run_study_A(df)
    b, results = run_study_B(df)
    summary = pd.concat([a, b], ignore_index=True)
    summary.to_csv("baseline_comparison_results.csv", index=False)

    pd.set_option("display.width", 160)
    print("=== Study A: stationary grid ===")
    print(a.to_string(index=False))
    print("\nStationary-grid MEAN Sharpe-like score per strategy:")
    a_means = a.groupby("variant")["score"].mean().reindex(VARIANTS)
    a_pnl = a.groupby("variant")["total_pnl"].mean().reindex(VARIANTS)
    for v in VARIANTS:
        print(f"  {v:<10} mean score = {a_means[v]:7.2f}   mean PnL = {a_pnl[v]:9.0f}")

    print("\n=== Study B: regime-shift stream ===")
    print(b.to_string(index=False))

    obj = results["objective"]["total_pnl"]
    fm = results["fillmodel"]["total_pnl"]
    st = results["static"]["total_pnl"]
    print("\n=== Headline gaps used on the CV (regime-shift stream) ===")
    print(f"  objective total PnL : {obj:,.0f}")
    print(f"  vs fill-model       : {fm:,.0f}  -> objective is {(obj/fm-1)*100:+.1f}% PnL")
    print(f"  vs static           : {st:,.0f}  -> objective is {(obj/st-1)*100:+.1f}% PnL")
    print(f"  objective score     : {results['objective']['score']:.2f}")
    print(f"  objective max drawdown: {results['objective']['max_drawdown']:.1f}")
    print("\nwrote -> baseline_comparison_results.csv")


if __name__ == "__main__":
    main()
