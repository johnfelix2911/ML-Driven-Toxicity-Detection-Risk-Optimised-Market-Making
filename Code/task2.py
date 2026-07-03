from typing import List
import os
import numpy as np
import pandas as pd

_DATA_PATH = os.environ.get(
    "TRADE_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_data.csv"),
)
_TAUS = [5, 10, 15, 20, 25, 30]
_df_cache = None


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(_DATA_PATH)
    return _df_cache


def _client_frame(client: str) -> pd.DataFrame:
    df = _load_data()
    sub = df[df["Name"] == client]
    if len(sub) == 0:
        raise ValueError(f"No trades found for client {client!r}")
    return sub

def expected_pnl(client: str, tau: List[int]) -> dict:
    sub = _client_frame(client)
    per_horizon: List[float] = []
    for t in tau:
        pnl = sub["Side"] * sub["Volume"] * (sub[f"M{t}"] - sub["Trade Price"])
        per_horizon.append(float(pnl.mean()))

    # Aggregate uses ALL six horizons with uniform weights (independent of `tau`).
    mids = sub[[f"M{t}" for t in _TAUS]].to_numpy()
    agg_per_trade = (
        sub["Side"].to_numpy() * sub["Volume"].to_numpy()
        * (mids.mean(axis=1) - sub["Trade Price"].to_numpy())
    )
    aggregate = float(agg_per_trade.mean())

    return {"per_horizon": per_horizon, "aggregate": aggregate}


def classify_client(client: str) -> str:
    agg = expected_pnl(client, _TAUS)["aggregate"]
    return "profitable" if agg >= 0 else "costly"


def min_half_spread(client: str) -> float:
    sub = _client_frame(client)
    mids = sub[[f"M{t}" for t in _TAUS]].to_numpy()
    base = (
        sub["Side"].to_numpy() * sub["Volume"].to_numpy()
        * (mids.mean(axis=1) - sub["M0"].to_numpy())
    ).mean()
    v_mean = sub["Volume"].mean()
    delta = -base / v_mean
    return float(max(0.0, delta))

def build_results_csv(path: str = "task2_results.csv") -> pd.DataFrame:
    df = _load_data()
    clients = sorted(df["Name"].unique())
    rows = []
    for c in clients:
        ep = expected_pnl(c, _TAUS)
        row = [c] + [round(v, 6) for v in ep["per_horizon"]]
        row.append(round(ep["aggregate"], 6))
        row.append(round(min_half_spread(c), 8))
        rows.append(row)
    cols = ["client"] + [f"tau={t}" for t in _TAUS] + ["agg_pnl", "delta_star"]
    res = pd.DataFrame(rows, columns=cols)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    res = build_results_csv()
    pd.set_option("display.width", 160)
    print(res.to_string(index=False))
    df = _load_data()
    for c in sorted(df["Name"].unique()):
        print(f"  {c}: {classify_client(c)}")