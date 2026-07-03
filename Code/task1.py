from typing import List
import os
import numpy as np
import pandas as pd

_DATA_PATH = os.environ.get(
    "TRADE_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_data.csv")
)
_TAUS = [5, 10, 15, 20, 25, 30]
_df_cache = None


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        df = pd.read_csv(_DATA_PATH)
        _df_cache = df
    return _df_cache


def _is_adverse(sub: pd.DataFrame, tau: int) -> pd.Series:
    mid_t = sub[f"M{tau}"]
    pnl_sign = sub["Side"] * (mid_t - sub["Trade Price"])   # V > 0 omitted
    return pnl_sign < 0

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    df = _load_data()
    sub = df[df["Name"] == client]
    if len(sub) == 0:
        raise ValueError(f"No trades found for client {client!r}")

    out: List[float] = []
    for t in tau:
        if f"M{t}" not in df.columns:
            raise ValueError(f"Horizon tau={t} not available in data.")
        out.append(float(_is_adverse(sub, t).mean() * 100.0))
    return out
    
def build_results_csv(path: str = "task1_results.csv") -> pd.DataFrame:
    df = _load_data()
    clients = sorted(df["Name"].unique())
    rows = []
    for c in clients:
        prof = adversity_profile(c, _TAUS)
        rows.append([c] + [round(v, 4) for v in prof])
    cols = ["client"] + [f"tau={t}" for t in _TAUS]
    res = pd.DataFrame(rows, columns=cols)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    res = build_results_csv()
    print(res.to_string(index=False))