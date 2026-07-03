from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import task3  # trained model M, features, and 60/20/20 date split

TAUS = [5, 10, 15, 20, 25, 30]
CLIENTS = ["A", "B", "C", "D", "E", "F"]
_THETAS = np.linspace(0.0, 1.0, 201)   # cutoff grid

def _horizon_table(tau: int) -> pd.DataFrame:
    """Return per-trade table with predicted p, realised pnl, client, split."""
    st = task3._ensure_trained()
    df, X, models, masks = st["df"], st["X"], st["models"], st["masks"]
    p = models[tau].predict_proba(X)[:, 1]
    pnl = (df["side"] * df["volume"] * (df[f"M{tau}"] - df["Trade Price"])).to_numpy()

    split = np.empty(len(df), dtype=object)
    split[masks["train"]] = "train"
    split[masks["validation"]] = "validation"
    split[masks["test"]] = "test"

    return pd.DataFrame({
        "client": df["Name"].to_numpy(),
        "p": p, "pnl": pnl, "split": split,
    })


def _pnl_curve(p: np.ndarray, pnl: np.ndarray, thetas=_THETAS) -> np.ndarray:
    order = np.argsort(p)
    p_s, pnl_s = p[order], pnl[order]
    cum = np.concatenate([[0.0], np.cumsum(pnl_s)])
    # number kept = count of p_s <= theta
    idx = np.searchsorted(p_s, thetas, side="right")
    return cum[idx]

def optimal_threshold(tau: int,
                      client: Optional[str] = None,
                      client_specific: bool = False,
                      thetas: np.ndarray = _THETAS) -> dict:
    tab = _horizon_table(tau)

    def _solve(sub):
        v = sub[sub["split"] == "validation"]
        t = sub[sub["split"] == "test"]
        curve = _pnl_curve(v["p"].to_numpy(), v["pnl"].to_numpy(), thetas)
        j = int(np.argmax(curve))
        th = float(thetas[j])
        val_pnl = float(curve[j])
        keep = t["p"].to_numpy() <= th
        test_pnl = float(t["pnl"].to_numpy()[keep].sum())
        return th, val_pnl, test_pnl

    if client_specific:
        thetas_d, vsum, tsum = {}, 0.0, 0.0
        for c in sorted(tab["client"].unique()):
            th, vp, tp = _solve(tab[tab["client"] == c])
            thetas_d[c] = th
            vsum += vp
            tsum += tp
        return {"theta": thetas_d, "validation_pnl": vsum, "test_pnl": tsum}

    if client is not None:
        th, vp, tp = _solve(tab[tab["client"] == client])
        return {"theta": th, "validation_pnl": vp, "test_pnl": tp}

    th, vp, tp = _solve(tab)
    return {"theta": th, "validation_pnl": vp, "test_pnl": tp}


def plot_pnl_vs_theta(taus: List[int] = TAUS,
                      path: str = "pnl_vs_theta.png",
                      thetas: np.ndarray = _THETAS) -> None:
    plt.figure(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(taus)))
    for color, tau in zip(cmap, taus):
        tab = _horizon_table(tau)
        v = tab[tab["split"] == "validation"]
        curve = _pnl_curve(v["p"].to_numpy(), v["pnl"].to_numpy(), thetas)
        j = int(np.argmax(curve))
        plt.plot(thetas, curve, color=color, label=f"τ={tau} (θ*={thetas[j]:.2f})")
        plt.scatter([thetas[j]], [curve[j]], color=color, s=35, zorder=5,
                    edgecolor="k", linewidths=0.5)
    plt.axhline(0, color="grey", lw=0.8, ls="--")
    plt.xlabel("Externalization cutoff  θ")
    plt.ylabel("Validation PnL  (externalize if p > θ)")
    plt.title("Validation PnL vs externalization threshold (global)")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def build_results_csv(path: str = "task4_results.csv") -> pd.DataFrame:
    rows = []
    for tau in TAUS:
        tab = _horizon_table(tau)
        for c in CLIENTS:
            sub = tab[tab["client"] == c]
            v = sub[sub["split"] == "validation"]
            t = sub[sub["split"] == "test"]
            curve = _pnl_curve(v["p"].to_numpy(), v["pnl"].to_numpy(), _THETAS)
            j = int(np.argmax(curve))
            th = float(_THETAS[j])
            keep = t["p"].to_numpy() <= th
            final_pnl = float(t["pnl"].to_numpy()[keep].sum())
            rows.append([c, tau, round(th, 4), round(final_pnl, 4)])
    res = pd.DataFrame(rows, columns=["client", "tau", "theta_star", "final_pnl"])
    res = res.sort_values(["client", "tau"]).reset_index(drop=True)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    task3.train_pipeline()
    plot_pnl_vs_theta()
    res = build_results_csv()
    pd.set_option("display.width", 140)
    print(res.to_string(index=False))
    print("\nGlobal optimum per horizon:")
    for tau in TAUS:
        g = optimal_threshold(tau)
        cs = optimal_threshold(tau, client_specific=True)
        print(f"  tau={tau:>2}: global theta*={g['theta']:.3f} "
              f"val={g['validation_pnl']:.1f} test={g['test_pnl']:.1f} | "
              f"client-specific val={cs['validation_pnl']:.1f} "
              f"test={cs['test_pnl']:.1f}")