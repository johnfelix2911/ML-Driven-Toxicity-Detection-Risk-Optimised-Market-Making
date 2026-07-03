from typing import List, Dict
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
)

warnings.filterwarnings("ignore")

_DATA_PATH = os.environ.get(
    "TRADE_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_data.csv"),
)
TAUS = [5, 10, 15, 20, 25, 30]
CLIENTS = ["A", "B", "C", "D", "E", "F"]

FEATURE_NAMES = [
    "client_A", "client_B", "client_C", "client_D", "client_E", "client_F",
    "side", "volume", "log_volume", "spread", "rel_spread",
    "tp_minus_mid", "signed_offset", "tod_frac", "realized_vol", "momentum",
]

_STATE: Dict = {}

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # chronological order is essential for causal rolling features
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["time"])
    df = df.sort_values("dt").reset_index(drop=True)

    # one-hot client
    for c in CLIENTS:
        df[f"client_{c}"] = (df["Name"] == c).astype(float)

    df["side"] = df["Side"].astype(float)
    df["volume"] = df["Volume"].astype(float)
    df["log_volume"] = np.log1p(df["Volume"].astype(float))
    df["spread"] = df["Spread"].astype(float)
    df["rel_spread"] = df["Spread"] / df["M0"]
    df["tp_minus_mid"] = df["Trade Price"] - df["M0"]
    df["signed_offset"] = df["side"] * df["tp_minus_mid"]

    # time of day fraction (per day, relative to that day's first/last trade)
    secs = df["dt"].dt.hour * 3600 + df["dt"].dt.minute * 60 + df["dt"].dt.second
    df["secs"] = secs
    g = df.groupby("Date")["secs"]
    lo, hi = g.transform("min"), g.transform("max")
    df["tod_frac"] = (df["secs"] - lo) / (hi - lo).replace(0, 1)

    # causal market-state features from the global mid series
    mid = df["M0"].to_numpy()
    ret = np.zeros_like(mid)
    ret[1:] = mid[1:] / mid[:-1] - 1.0
    df["ret"] = ret
    # realized vol over last 20 returns, momentum over last 5; SHIFT by 1 so the
    # current trade only sees information strictly before it.
    rser = pd.Series(ret)
    df["realized_vol"] = rser.rolling(20, min_periods=5).std().shift(1).fillna(0.0)
    df["momentum"] = rser.rolling(5, min_periods=1).sum().shift(1).fillna(0.0)

    return df


def _labels(df: pd.DataFrame, tau: int) -> np.ndarray:
    return ((df["side"] * (df[f"M{tau}"] - df["Trade Price"])) < 0).astype(int).to_numpy()


def _date_split(df: pd.DataFrame):
    dates = sorted(df["Date"].unique())
    n = len(dates)
    n_tr = int(round(0.60 * n))
    n_va = int(round(0.20 * n))
    tr_d = set(dates[:n_tr])
    va_d = set(dates[n_tr:n_tr + n_va])
    te_d = set(dates[n_tr + n_va:])
    tr = df["Date"].isin(tr_d).to_numpy()
    va = df["Date"].isin(va_d).to_numpy()
    te = df["Date"].isin(te_d).to_numpy()
    return tr, va, te

def train_pipeline(data_path: str = _DATA_PATH) -> Dict:
    raw = pd.read_csv(data_path)
    df = _build_features(raw)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    tr, va, te = _date_split(df)

    models, metrics_rows = {}, {sp: [] for sp in ["train", "validation", "test"]}
    val_proba, val_mask = {}, {"train": tr, "validation": va, "test": te}

    for tau in TAUS:
        y = _labels(df, tau)
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=200, random_state=42,
        )
        clf.fit(X[tr], y[tr])
        models[tau] = clf

        for sp, mask in val_mask.items():
            p = clf.predict_proba(X[mask])[:, 1]
            yhat = (p >= 0.5).astype(int)
            ytrue = y[mask]
            metrics_rows[sp].append({
                "accuracy": accuracy_score(ytrue, yhat),
                "precision": precision_score(ytrue, yhat, zero_division=0),
                "recall": recall_score(ytrue, yhat, zero_division=0),
                "log_loss": log_loss(ytrue, p, labels=[0, 1]),
            })

    # average metrics across horizons
    metrics = {}
    for sp in ["train", "validation", "test"]:
        dfm = pd.DataFrame(metrics_rows[sp])
        metrics[sp] = dfm.mean().to_dict()

    _STATE.update({
        "df": df, "X": X, "models": models, "masks": val_mask,
        "metrics": metrics, "feature_names": FEATURE_NAMES,
    })
    return _STATE

def _ensure_trained():
    if "models" not in _STATE:
        train_pipeline()
    return _STATE

def _vectorize(features) -> np.ndarray:
    """Accept a dict (named features) or an ordered iterable -> 1x16 array."""
    if isinstance(features, dict):
        vec = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
    else:
        vec = list(map(float, features))
        if len(vec) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {len(vec)}."
            )
    return np.asarray(vec, dtype=float).reshape(1, -1)


def predict_adversity(*args, **kwargs) -> float:
    _ensure_trained()

    tau = kwargs.pop("tau", None)
    features = kwargs.pop("features", None)
    if features is None and args:
        features = args[0]
        if len(args) > 1 and tau is None:
            tau = args[1]
    if tau is None:
        raise ValueError("`tau` must be provided.")
    tau = int(tau)
    if tau not in _STATE["models"]:
        raise ValueError(f"No model for tau={tau}; available {TAUS}.")

    if features is None:
        # build a feature dict from convenience kwargs
        feats = {}
        cl = kwargs.get("client")
        if cl is not None:
            feats[f"client_{cl}"] = 1.0
        side = float(kwargs.get("side", 0.0))
        vol = float(kwargs.get("volume", 0.0))
        m0 = float(kwargs.get("m0", kwargs.get("M0", 0.0)))
        tp = float(kwargs.get("trade_price", kwargs.get("TP", m0)))
        spread = float(kwargs.get("spread", 0.0))
        feats.update({
            "side": side, "volume": vol, "log_volume": np.log1p(max(vol, 0.0)),
            "spread": spread, "rel_spread": (spread / m0) if m0 else 0.0,
            "tp_minus_mid": tp - m0, "signed_offset": side * (tp - m0),
            "tod_frac": float(kwargs.get("tod_frac", 0.0)),
            "realized_vol": float(kwargs.get("realized_vol", 0.0)),
            "momentum": float(kwargs.get("momentum", 0.0)),
        })
        features = feats

    X1 = _vectorize(features)
    return float(_STATE["models"][tau].predict_proba(X1)[0, 1])


def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    _ensure_trained()
    m = _STATE["metrics"]
    out = pd.DataFrame(
        [m["train"], m["validation"], m["test"]],
        index=["train", "validation", "test"],
    )[["accuracy", "precision", "recall", "log_loss"]]
    return out

def build_results_csv(path: str = "task3_results.csv") -> pd.DataFrame:
    m = compute_metrics()
    res = m.reset_index().rename(columns={"index": "split"})
    res.to_csv(path, index=False)
    with open("task3_features.txt", "w") as f:
        f.write("Task 3 - Feature vector ordering (index : name)\n")
        f.write("=" * 46 + "\n")
        for i, name in enumerate(FEATURE_NAMES):
            f.write(f"{i:>2} : {name}\n")
    return res


if __name__ == "__main__":
    train_pipeline()
    res = build_results_csv()
    pd.set_option("display.width", 140)
    print(res.round(5).to_string(index=False))