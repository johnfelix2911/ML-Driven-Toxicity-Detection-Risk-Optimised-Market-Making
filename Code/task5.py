from typing import Dict, Tuple, List, Optional
from collections import deque
import os
import numpy as np
import pandas as pd

CMIN = 0.5            # delta >= cmin * sigma   (=> x >= 0.5)
DELTA_MAX = 0.005     # 50 bps of mid (fractional)

BETA_B0 = 5.84
BETA_B1 = -12.95
BETA_CLIP = (-3.0, 1.5)        # bound extrapolation outside the observed alpha range

X_FLOOR = CMIN                 # lower bound on the normalised half-spread x
X_SYM_MAX = 8.0                # upper bound on x before the delta_max clip
X_STATIC = 1.10                # fixed multiple used by the non-adaptive baseline
GAMMA0 = 1.0                   # prior gamma before enough fills are observed
KAPPA = 0.70                   # inventory-skew strength (sigma units, saturated)
K_ETA = 4.00                   # end-of-day urgency multiplier
I_SCALE = 1500.0               # inventory normalisation (~ a day of one-sided flow)

CTRL_WIN = 4000                # rolling window of (x, fill) observations
CTRL_PERIOD = 300              # arrivals between updates
CTRL_MINBIN = 30               # min samples in a quantile bin to use it
GAMMA_SMOOTH = 0.5             # EWMA smoothing of the gamma estimate
DITHER = 0.08                  # +/- multiplicative probe (gamma id. + SPSA)
SPSA_BETA = 0.02               # EWMA weight of the realised-PnL gradient
SPSA_LR = 0.15                 # step applied to the residual correction xc
SPSA_SCALE = 0.02              # EWMA weight of the reward-scale normaliser
XC_CLIP = (-2.0, 2.0)          # bound on the SPSA residual correction
SIG_EMA = 0.02                 # EWMA weight for the volatility-shock baseline
VOLSHOCK_CAP = 2.0             # cap on sigma / sigma_ema used for the guard
VOLSHOCK_POW = 0.5             # exponent of the volatility-shock widening
DAY_DROP = 0.30                # eta drop that signals a new trading day

_RS: Dict = {}


def beta_edge(alpha: float) -> float:
    """Calibrated realised edge in sigma-units as a function of adversity alpha."""
    b = BETA_B0 + BETA_B1 * float(alpha)
    return float(np.clip(b, BETA_CLIP[0], BETA_CLIP[1]))


def reset_quote_state(mode: str = "objective", spsa: bool = False) -> None:
    _RS.clear()
    _RS.update({
        "mode": mode, "spsa": bool(spsa),
        "prev_I": None, "prev_eta": None, "prev_sigma": None, "prev_x": None,
        "obs_x": deque(maxlen=CTRL_WIN), "obs_f": deque(maxlen=CTRL_WIN),
        "step": 0, "gamma_hat": None, "xc": 0.0,
        "g_ewma": 0.0, "rew_scale": None, "last_dither": 1.0,
        "sigma_ema": None, "dither": 1.0, "x_last": X_STATIC,
    })


def _ensure_state() -> None:
    if not _RS:
        reset_quote_state(mode="objective", spsa=False)


def quote(inventory: float, sigma: float,
          alpha: float, eta: float) -> Tuple[float, float]:
    _ensure_state()
    st = _RS
    mode = st["mode"]
    sigma = float(max(sigma, 1e-12))
    alpha = float(min(max(alpha, 0.0), 1.0))
    eta = float(min(max(eta, 0.0), 1.0))

    adapt = mode in ("objective", "fillmodel")

    # ----- 1. learn gamma from the previous arrival's fill (inventory delta) - #
    if adapt and st["prev_I"] is not None:
        new_day = (st["prev_eta"] is not None) and (eta < st["prev_eta"] - DAY_DROP)
        if not new_day:
            dI = inventory - st["prev_I"]
            filled = 1.0 if abs(dI) > 1e-9 else 0.0
            st["obs_x"].append(st["prev_x"])      # exposed x  (regressor)
            st["obs_f"].append(filled)            # fill outcome
            st["step"] += 1
            if st["step"] % CTRL_PERIOD == 0 and len(st["obs_x"]) >= CTRL_PERIOD:
                g = _estimate_gamma(np.asarray(st["obs_x"]),
                                    np.asarray(st["obs_f"]))
                if g is not None and g > 1e-3:
                    st["gamma_hat"] = g if st["gamma_hat"] is None else \
                        (1 - GAMMA_SMOOTH) * st["gamma_hat"] + GAMMA_SMOOTH * g
                # apply the accumulated realised-PnL gradient to the residual xc
                if mode == "objective" and st["spsa"]:
                    st["xc"] = float(np.clip(st["xc"] + SPSA_LR * st["g_ewma"],
                                             XC_CLIP[0], XC_CLIP[1]))

    # ----- 2. target normalised half-spread x (the objective optimum) -------- #
    g_eff = st["gamma_hat"] if st["gamma_hat"] is not None else GAMMA0
    if mode == "static":
        x_sym = X_STATIC
    elif mode == "fillmodel":
        x_sym = 1.0 / g_eff                                   # old: fill-model opt
    else:  # objective
        x_sym = 1.0 / g_eff - beta_edge(alpha) + st["xc"]     # eq. (*) + SPSA
    x_sym = float(np.clip(x_sym, X_FLOOR, X_SYM_MAX))
    st["x_last"] = x_sym

    # ----- 3. volatility-shock guard + probe dither -------------------------- #
    if st["sigma_ema"] is None:
        st["sigma_ema"] = sigma
    else:
        st["sigma_ema"] = (1 - SIG_EMA) * st["sigma_ema"] + SIG_EMA * sigma
    if adapt:
        shock = float(np.clip(sigma / max(st["sigma_ema"], 1e-12),
                              1.0, VOLSHOCK_CAP)) ** VOLSHOCK_POW
        st["dither"] = -st["dither"]
        dith = 1.0 + DITHER * st["dither"]
    else:
        shock, dith = 1.0, 1.0

    # ----- 4. compose quote: symmetric base +/- inventory skew --------------- #
    h = sigma * x_sym * dith * shock
    s = sigma * KAPPA * np.tanh(inventory / I_SCALE) * (1.0 + K_ETA * eta)
    lo = CMIN * sigma
    delta_bid = float(np.clip(h + s, lo, DELTA_MAX))
    delta_ask = float(np.clip(h - s, lo, DELTA_MAX))

    # ----- 5. remember for next call ---------------------------------------- #
    st["prev_I"] = float(inventory)
    st["prev_eta"] = eta
    st["prev_sigma"] = sigma
    st["prev_x"] = x_sym * dith * shock        # the x actually exposed (regressor)
    st["last_dither"] = st["dither"] if adapt else 0.0
    return delta_bid, delta_ask


def observe(reward_frac: float) -> None:
    if not _RS or not _RS.get("spsa"):
        return
    st = _RS
    r = float(reward_frac)
    a = abs(r)
    st["rew_scale"] = a if st["rew_scale"] is None else \
        (1 - SPSA_SCALE) * st["rew_scale"] + SPSA_SCALE * a
    scale = max(st["rew_scale"], 1e-12)
    rn = r / scale
    st["g_ewma"] = (1 - SPSA_BETA) * st["g_ewma"] + SPSA_BETA * (st["last_dither"] * rn)


def _estimate_gamma(x: np.ndarray, f: np.ndarray) -> Optional[float]:
    if len(x) < CTRL_PERIOD:
        return None
    edges = np.unique(np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(edges) < 3:
        return None
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    mx, ly = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= CTRL_MINBIN:
            fr = f[sel].mean()
            if 0.0 < fr < 1.0:
                mx.append(x[sel].mean()); ly.append(np.log(fr))
    if len(mx) < 2:
        return None
    slope = np.polyfit(np.asarray(mx), np.asarray(ly), 1)[0]
    return float(-slope)

_DATA_PATH = os.environ.get(
    "TRADE_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_data.csv"),
)
_TAUS = [5, 10, 15, 20, 25, 30]


def _prepare_stream(data_path: str = _DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["time"])
    df = df.sort_values("dt").reset_index(drop=True)

    mid = df["M0"].to_numpy()
    ret = np.zeros_like(mid)
    ret[1:] = mid[1:] / mid[:-1] - 1.0
    sig = pd.Series(ret).pow(2).rolling(20, min_periods=5).mean().pow(0.5)
    df["sigma"] = sig.shift(1).bfill().fillna(1e-4).clip(lower=1e-6).to_numpy()

    g = df.groupby("Date")
    lo = g["dt"].transform("min")
    span = (g["dt"].transform("max") - lo).dt.total_seconds().replace(0, 1)
    df["eta"] = ((df["dt"] - lo).dt.total_seconds() / span).to_numpy()

    df["alpha"] = _alpha_scores(df)
    df["mid_close"] = df[[f"M{t}" for t in _TAUS]].mean(axis=1).to_numpy()
    return df


def _alpha_scores(df: pd.DataFrame) -> np.ndarray:
    try:
        import task3
        st = task3._ensure_trained()
        feat = task3._build_features(df)
        Xf = feat[task3.FEATURE_NAMES].to_numpy(float)
        probs = np.zeros(len(df))
        for tau in _TAUS:
            probs += st["models"][tau].predict_proba(Xf)[:, 1]
        return probs / len(_TAUS)
    except Exception:
        rate = {}
        for c in df["Name"].unique():
            sub = df[df["Name"] == c]
            r = np.mean([
                ((sub["Side"] * (sub[f"M{t}"] - sub["Trade Price"])) < 0).mean()
                for t in _TAUS
            ])
            rate[c] = r
        return df["Name"].map(rate).to_numpy()


def _metrics(daily: Dict) -> Dict:
    pnl = np.array([daily[d] for d in sorted(daily)])
    total = float(pnl.sum())
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    floor = max(abs(total) / max(len(pnl), 1) * 0.05, 1.0)
    score = total / max(std, floor)
    cum = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
    return {"total_pnl": total, "daily_std": std, "score": score,
            "max_drawdown": dd, "n_days": len(pnl), "daily_pnl": pnl}


def _run(df: pd.DataFrame, lam, gam, phi, tox=1.0, seed: int = 7,
         mode: str = "objective", spsa: bool = False, collect_x: bool = False,
         reg_idx: Optional[np.ndarray] = None, n_reg: int = 0) -> Dict:
    n = len(df)

    def _arr(x):
        return x if isinstance(x, np.ndarray) else np.full(n, float(x))
    lam_a, gam_a, phi_a, tox_a = _arr(lam), _arr(gam), _arr(phi), _arr(tox)

    side = df["Side"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    m0 = df["M0"].to_numpy(float)
    midc = df["mid_close"].to_numpy(float)
    sig = df["sigma"].to_numpy(float)
    alp = df["alpha"].to_numpy(float)
    eta = df["eta"].to_numpy(float)
    day = df["Date"].to_numpy()
    u = np.random.default_rng(seed).random(n)

    reset_quote_state(mode=mode, spsa=spsa)
    daily, I, cur_day, day_sig = {}, 0.0, None, []
    x_trace, last_phi = [], 0.0
    reg_pnl = np.zeros(max(n_reg, 1)) if reg_idx is not None else None
    last_reg = 0

    def _close(d):
        nonlocal I, day_sig
        if d is None:
            return
        sd = float(np.mean(day_sig)) if day_sig else 0.0
        pen = last_phi * (I ** 2) * sd
        daily[d] = daily.get(d, 0.0) - pen
        if reg_pnl is not None:
            reg_pnl[last_reg] -= pen
        if collect_x:
            x_trace.append(_RS.get("x_last", X_STATIC))
        I, day_sig = 0.0, []

    for i in range(n):
        d = day[i]
        if d != cur_day:
            _close(cur_day)
            cur_day = d
            daily.setdefault(d, 0.0)
        day_sig.append(sig[i])
        last_phi = phi_a[i]
        if reg_idx is not None:
            last_reg = int(reg_idx[i])

        db, da = quote(I, sig[i], alp[i], eta[i])
        delta = db if side[i] > 0 else da
        p_fill = lam_a[i] * np.exp(-gam_a[i] * delta / sig[i])
        filled = u[i] < p_fill
        reward_frac = 0.0
        if filled:
            captured = vol[i] * delta * m0[i]
            realised_edge = tox_a[i] * side[i] * vol[i] * (midc[i] - m0[i])
            gain = captured + realised_edge
            daily[d] += gain
            if reg_pnl is not None:
                reg_pnl[last_reg] += gain
            I += side[i] * vol[i]
            # realised fractional PnL per unit volume (what a live desk observes)
            reward_frac = delta + tox_a[i] * side[i] * (midc[i] - m0[i]) / m0[i]
        if spsa:
            observe(reward_frac)
    _close(cur_day)

    out = _metrics(daily)
    out["x_trace"] = x_trace
    if reg_pnl is not None:
        out["regime_pnl"] = reg_pnl
    return out


def backtest(df: pd.DataFrame, lam: float, gamma: float, phi: float,
             seed: int = 7, mode: str = "objective", spsa: bool = False) -> Dict:
    return _run(df, lam, gamma, phi, tox=1.0, seed=seed, mode=mode, spsa=spsa)


def _regime_arrays(df: pd.DataFrame, schedule: List[dict]) -> Dict[str, np.ndarray]:
    n = len(df)
    lam = np.empty(n); gam = np.empty(n); phi = np.empty(n); tox = np.empty(n)
    idx = np.zeros(n, dtype=int)
    start, bounds = 0, []
    for k, seg in enumerate(schedule):
        end = n if k == len(schedule) - 1 else min(n, start + int(round(seg["frac"] * n)))
        lam[start:end] = seg["lam"]; gam[start:end] = seg["gam"]
        phi[start:end] = seg["phi"]; tox[start:end] = seg["tox"]
        idx[start:end] = k
        bounds.append((start, end))
        start = end
    return {"lam": lam, "gam": gam, "phi": phi, "tox": tox,
            "idx": idx, "bounds": bounds}


def validate_quote(data_path: str = _DATA_PATH,
                   save_fig: str = "task5_validation.png") -> pd.DataFrame:
    df = _prepare_stream(data_path)
    rows: List[dict] = []

    # ----------------------- A. stationary grid ----------------------------- #
    grid = [(0.5, 0.7, 1e-6), (0.5, 1.0, 1e-6), (0.6, 1.5, 1e-6),
            (0.5, 2.0, 1e-6), (0.7, 0.8, 1e-5)]
    for (lam, gam, phi) in grid:
        for name, md in [("objective", "objective"), ("fillmodel", "fillmodel"),
                         ("static", "static")]:
            r = backtest(df, lam, gam, phi, seed=7, mode=md)
            rows.append({"study": "A-stationary", "variant": name,
                         "lam": lam, "gam": gam, "phi": phi,
                         "total_pnl": round(r["total_pnl"], 1),
                         "score": round(r["score"], 2),
                         "max_dd": round(r["max_drawdown"], 1)})

    # ----------------------- B. regime-shift stream ------------------------- #
    schedule = [
        {"frac": 0.34, "lam": 0.80, "gam": 0.60, "phi": 1e-6, "tox": 1.00},
        {"frac": 0.33, "lam": 0.35, "gam": 2.80, "phi": 5e-4, "tox": 0.40},
        {"frac": 0.33, "lam": 0.60, "gam": 1.30, "phi": 5e-6, "tox": 1.00},
    ]
    reg = _regime_arrays(df, schedule)
    nreg = len(schedule)
    variants = [("static", "static", False), ("fillmodel", "fillmodel", False),
                ("objective", "objective", False), ("objective+SPSA", "objective", True)]
    reg_results = {}
    for name, md, sp in variants:
        r = _run(df, reg["lam"], reg["gam"], reg["phi"], reg["tox"], seed=7,
                 mode=md, spsa=sp, collect_x=(name in ("objective", "objective+SPSA")),
                 reg_idx=reg["idx"], n_reg=nreg)
        reg_results[name] = r
        rows.append({"study": "B-regime-shift", "variant": name,
                     "lam": "shift", "gam": "shift", "phi": "shift",
                     "total_pnl": round(r["total_pnl"], 1),
                     "score": round(r["score"], 2),
                     "max_dd": round(r["max_drawdown"], 1)})

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 170)
    print("=== Task 5 validation: objective-optimal vs fill-model target ===\n")
    a = summary[summary["study"] == "A-stationary"]
    print("A. Stationary grid:")
    print(a.to_string(index=False))
    for v in ("objective", "fillmodel", "static"):
        s = a[a.variant == v]
        print(f"  {v:<10} mean score={s.score.mean():7.2f}  mean PnL={s.total_pnl.mean():9.0f}")

    print("\nB. Regime-shift stream:")
    b = summary[summary["study"] == "B-regime-shift"]
    print(b.to_string(index=False))
    rn = ["R1 easy(g0.6)", "R2 hard(g2.8,tox)", "R3 med(g1.3)"]
    print("\n  Per-regime PnL:")
    print("    {:<15}".format("variant") + "".join(f"{x:>20}" for x in rn))
    for name, _, _ in variants:
        rp = reg_results[name].get("regime_pnl", np.zeros(nreg))
        print("    {:<15}".format(name) + "".join(f"{v:>20.0f}" for v in rp))

    obj, fm, stat = (reg_results["objective"]["total_pnl"],
                     reg_results["fillmodel"]["total_pnl"],
                     reg_results["static"]["total_pnl"])
    print(f"\n  objective vs fill-model:  PnL {obj:.0f} vs {fm:.0f} ({(obj/fm-1)*100:+.1f}%),"
          f"  score {reg_results['objective']['score']:.2f} vs {reg_results['fillmodel']['score']:.2f}")
    print(f"  objective vs static:      PnL {obj:.0f} vs {stat:.0f} ({(obj/stat-1)*100:+.1f}%),"
          f"  score {reg_results['objective']['score']:.2f} vs {reg_results['static']['score']:.2f}")

    # ----------------------- figure ----------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ndays = reg_results["objective"]["n_days"]
        bnds = [int(round(b2[1] / len(df) * ndays)) for b2 in reg["bounds"][:-1]]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 5.4),
                                       gridspec_kw={"height_ratios": [1.9, 1]})
        col = {"static": "C7", "fillmodel": "C1",
               "objective": "C0", "objective+SPSA": "C2"}
        for name, _, _ in variants:
            r = reg_results[name]
            ax1.plot(np.cumsum(r["daily_pnl"]), lw=1.9, color=col[name],
                     label=f"{name} (PnL={r['total_pnl']:.0f}, score={r['score']:.1f})")
        for x in bnds:
            ax1.axvline(x, color="grey", ls="--", lw=0.9, alpha=0.7)
        ax1.set_title("B. Cumulative net PnL under undisclosed regime shifts "
                      "(dashed = regime boundary)")
        ax1.set_xlabel("trading day"); ax1.set_ylabel("cumulative net PnL")
        ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3)

        ax2.plot(reg_results["objective"]["x_trace"], lw=1.7, color="C0",
                 label="objective x*")
        ax2.plot(reg_results["objective+SPSA"]["x_trace"], lw=1.7, color="C2",
                 label="objective+SPSA x*")
        for x in bnds:
            ax2.axvline(x, color="grey", ls="--", lw=0.9, alpha=0.7)
        ax2.axhline(CMIN, color="k", lw=0.7, ls=":")
        ax2.set_title("Target half-spread x*(t) = 1/gamma - beta(alpha): "
                      "tight for benign volume, wide in the toxic regime")
        ax2.set_xlabel("trading day"); ax2.set_ylabel("x* (end of day)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(save_fig, dpi=140); plt.close()
        print(f"\nsaved figure -> {save_fig}")
    except Exception as e:
        print("figure skipped:", e)

    return summary


if __name__ == "__main__":
    reset_quote_state(mode="objective", spsa=False)
    print("smoke test (objective mode, isolated calls):")
    for I in (-3000, 0, 3000):
        for al in (0.35, 0.55):
            db, da = quote(I, 2e-4, al, 0.9)
            lo = CMIN * 2e-4
            ok = lo - 1e-12 <= db <= DELTA_MAX and lo - 1e-12 <= da <= DELTA_MAX
            print(f"  I={I:>6} alpha={al} -> bid={db:.6f} ask={da:.6f} ok={ok}")
    print()
    validate_quote()