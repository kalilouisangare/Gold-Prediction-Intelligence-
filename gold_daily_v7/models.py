from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config

BASELINES = ("naive", "drift")


def registry(cfg: Config) -> dict:
    reg = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=cfg.ridge_alpha))}
    if cfg.use_xgb:
        try:
            from xgboost import XGBRegressor
            reg["xgb"] = lambda: XGBRegressor(**cfg.xgb_params)
            reg["blend"] = None                      # composite ridge + xgb
        except ImportError:
            print("[MODÈLES] xgboost absent : ridge seul.")
    if cfg.use_mlp:
        from sklearn.neural_network import MLPRegressor
        reg["mlp"] = lambda: make_pipeline(StandardScaler(), MLPRegressor(**cfg.mlp_params))
        deep = {**cfg.mlp_params, "hidden_layer_sizes": (64, 32, 16)}
        reg["mlp_deep"] = lambda: make_pipeline(StandardScaler(), MLPRegressor(**deep))
    return reg


def candidates(reg: dict) -> list[str]:
    return [k for k in reg if k not in BASELINES]


def make_dataset(raw: pd.DataFrame, feats: pd.DataFrame, h: int):
    positions = pd.Series(np.arange(len(raw), dtype=int), index=raw.index)
    target = np.log(raw["gold_Close"].shift(-h) / raw["gold_Close"]).rename("y")
    end = positions.shift(-h).rename("target_end_pos")
    frame = feats.join(target).join(raw["gold_Close"].rename("p0")).join(end)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    cols = list(feats.columns)
    frame = frame.dropna(subset=cols + ["y", "p0", "target_end_pos"])
    meta = frame[["p0", "target_end_pos"]].copy()
    meta["target_end_pos"] = meta["target_end_pos"].astype(int)
    return frame[cols].astype(float), frame["y"].astype(float), meta


def drift_series(raw: pd.DataFrame, h: int, cfg: Config) -> pd.Series:
    r = np.log(raw["gold_Close"].astype(float)).diff()
    return (r.rolling(cfg.drift_window, min_periods=30).mean() * h).rename("drift")


def _predict(name, reg, Xtr, ytr, Xte):
    if name == "blend":
        return (_predict("ridge", reg, Xtr, ytr, Xte) + _predict("xgb", reg, Xtr, ytr, Xte)) / 2
    return reg[name]().fit(Xtr, ytr).predict(Xte)


def fit_predict(Xtr, ytr, Xte, reg) -> dict:
    out = {"naive": np.zeros(len(Xte), dtype=float)}
    for name in reg:
        out[name] = np.asarray(_predict(name, reg, Xtr, ytr, Xte), dtype=float)
    return out


def walk_forward(raw, feats, h, cfg: Config, reg):
    X, y, meta = make_dataset(raw, feats, h)
    required = cfg.n_splits * cfg.test_size + cfg.min_train_rows
    if len(X) < required:
        raise RuntimeError(f"H={h} : {len(X)} lignes utilisables, {required} requises.")

    raw_pos = pd.Series(np.arange(len(raw), dtype=int), index=raw.index)
    fpos = raw_pos.reindex(X.index).to_numpy(dtype=int)
    first = len(raw) - cfg.n_splits * cfg.test_size
    idx_parts, pred_parts = [], {}

    for fold in range(cfg.n_splits):
        lo = first + fold * cfg.test_size
        hi = lo + cfg.test_size
        te = np.flatnonzero((fpos >= lo) & (fpos < hi))
        tr = np.flatnonzero((fpos < lo) & (meta["target_end_pos"].to_numpy() < lo))
        if len(te) < int(cfg.test_size * 0.80):
            raise RuntimeError(f"H={h} fold {fold+1} : {len(te)} observations testables.")
        if len(tr) < cfg.min_train_rows:
            raise RuntimeError(f"H={h} fold {fold+1} : {len(tr)} lignes d'entraînement.")
        preds = fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], reg)
        idx_parts.append(X.index[te])
        for k, v in preds.items():
            pred_parts.setdefault(k, []).append(v)

    idx = pd.DatetimeIndex(np.concatenate([np.asarray(i, dtype="datetime64[ns]") for i in idx_parts]))
    preds = {k: pd.Series(np.concatenate(v), index=idx, name=k) for k, v in pred_parts.items()}
    preds["drift"] = drift_series(raw, h, cfg).reindex(idx)

    y_oos, p0 = y.loc[idx], meta.loc[idx, "p0"]
    scale = (feats.loc[idx, "vol_ewma"] * np.sqrt(h)).clip(lower=1e-8)
    if not np.isfinite(y_oos.to_numpy()).all() or not np.isfinite(scale.to_numpy()).all():
        raise RuntimeError("Valeurs non finies dans l'OOS.")
    return y_oos, preds, scale, p0


def boot_pvalue(d, cfg: Config) -> float:
    v = np.asarray(d, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 30:
        return float("nan")
    block = max(1, min(int(cfg.boot_block), n))
    obs = float(v.mean())
    centered = v - obs
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(cfg.seed)
    means = np.empty(cfg.n_boot)
    for i in range(cfg.n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        means[i] = np.concatenate([centered[s:s + block] for s in starts])[:n].mean()
    return float((1 + np.sum(means >= obs)) / (cfg.n_boot + 1))


def score_models(y, preds, p0, cfg: Config, drift=None) -> pd.DataFrame:
    actual = p0 * np.exp(y)
    naive_err = y.abs()
    base_naive = float(naive_err.mean())
    drift_err = None if drift is None else (y - drift).abs()
    base_drift = float("nan") if drift_err is None else float(drift_err.mean())
    rows = []
    for name, mu in preds.items():
        err = (y - mu).abs()
        mae = float(err.mean())
        is_n, is_d = name == "naive", name == "drift"
        rows.append({
            "modele": name, "MAE_log": mae,
            "MAE_USD": float((actual - p0 * np.exp(mu)).abs().mean()),
            "gain_vs_naive_%": np.nan if is_n or base_naive <= 0 else 100 * (1 - mae / base_naive),
            "p_vs_naive": np.nan if is_n else boot_pvalue((naive_err - err).to_numpy(), cfg),
            "gain_vs_drift_%": (np.nan if is_d or drift_err is None or
                                not np.isfinite(base_drift) or base_drift <= 0
                                else 100 * (1 - mae / base_drift)),
            "p_vs_drift": np.nan if is_d or drift_err is None
                          else boot_pvalue((drift_err - err).to_numpy(), cfg),
            "direction_%": np.nan if is_n else 100 * float((np.sign(mu) == np.sign(y)).mean()),
        })
    return pd.DataFrame(rows).round(4)


def select_model(table: pd.DataFrame, cfg: Config, n_candidates: int) -> str:
    thr = cfg.alpha_selection / max(n_candidates, 1)
    ok = table[(~table["modele"].isin(BASELINES))
               & (table["gain_vs_naive_%"] > 0) & (table["p_vs_naive"] < thr)
               & (table["gain_vs_drift_%"] > 0) & (table["p_vs_drift"] < thr)]
    return "naive" if ok.empty else str(ok.sort_values("MAE_log").iloc[0]["modele"])


def fit_one(name, X, y, x_last, reg, raw, h, last_date, cfg: Config) -> float:
    if name == "naive":
        return 0.0
    if name == "drift":
        s = drift_series(raw, h, cfg)
        v = float(s.loc[last_date]) if last_date in s.index else float(s.iloc[-1])
        return v if np.isfinite(v) else 0.0
    return float(_predict(name, reg, X, y, x_last)[0])