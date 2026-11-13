from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def build(model: pd.DataFrame, cfg: Config, masked: dict, bad_bars: list) -> tuple[pd.DataFrame, pd.Series]:
    g = model.copy()

    # quarantaine RÉELLE des barres OHLC incohérentes : les features de range
    # deviennent inconnues, au lieu d'alimenter le modèle avec un range négatif.
    if bad_bars and {"gold_High", "gold_Low"}.issubset(g.columns):
        g.loc[pd.DatetimeIndex(bad_bars), ["gold_High", "gold_Low"]] = np.nan

    c = g["gold_Close"].astype(float)
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=g.index)

    for k in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret_{k}"] = logc.diff(k)
    for w in (5, 20, 50, 200):
        f[f"dist_ma{w}"] = c / c.rolling(w).mean() - 1

    f["vol_5"] = r1.rolling(5).std()
    f["vol_20"] = r1.rolling(20).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"].replace(0, np.nan)
    f["vol_ewma"] = np.sqrt(r1.pow(2).ewm(alpha=0.06, adjust=False).mean())

    if {"gold_High", "gold_Low"}.issubset(g.columns):
        rng = (g["gold_High"] - g["gold_Low"]).replace(0, np.nan)
        f["range_hl"] = rng / c
        f["close_pos"] = (c - g["gold_Low"]) / rng
    if "gold_Open" in g.columns:
        f["gap_open"] = g["gold_Open"] / c.shift(1) - 1
    if "gold_Volume" in g.columns:
        vol = g["gold_Volume"].replace(0, np.nan)
        f["vol_rel"] = vol / vol.rolling(20).mean().replace(0, np.nan)

    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi = rsi.where(~((dn == 0) & (up > 0)), 100)
    rsi = rsi.where(~((dn == 0) & (up == 0)), 50)
    f["rsi_14"] = rsi
    f["mom_drift"] = r1.rolling(cfg.drift_window, min_periods=30).mean()

    for name in ("dxy", "oil", "silver", "vix"):
        col = f"{name}_Close"
        if col not in g.columns:
            continue
        s = g[col].ffill(limit=cfg.external_ffill_limit)
        ls = np.log(s.where(s > 0))
        f[f"{name}_ret_1"] = ls.diff().shift(cfg.external_lag)
        f[f"{name}_ret_5"] = ls.diff(5).shift(cfg.external_lag)
    if "us10y_Close" in g.columns:
        s = g["us10y_Close"].ffill(limit=cfg.external_ffill_limit)
        f["us10y_chg_1"] = s.diff().shift(cfg.external_lag)
        f["us10y_chg_5"] = s.diff(5).shift(cfg.external_lag)

    # remasquage : ni la valeur inconnue, ni son ffill, ni le rendement qui la traverse
    if masked:
        pos_of = {dt: i for i, dt in enumerate(g.index)}
        stems = {f"{n}_Close": (f"{n}_ret_1", f"{n}_ret_5") for n in ("dxy", "oil", "silver", "vix")}
        stems["us10y_Close"] = ("us10y_chg_1", "us10y_chg_5")
        for col, dates in masked.items():
            pair = stems.get(col)
            if pair is None:
                continue
            for dt in dates:
                pos = pos_of.get(dt)
                if pos is None:
                    continue
                for lag, stem in ((1, pair[0]), (5, pair[1])):
                    if stem not in f.columns:
                        continue
                    hi = min(pos + lag + cfg.external_ffill_limit + cfg.external_lag, len(f) - 1)
                    f.iloc[pos:hi + 1, f.columns.get_loc(stem)] = np.nan

    f["dow_sin"] = np.sin(2 * np.pi * g.index.day_of_week / 5)
    f["dow_cos"] = np.cos(2 * np.pi * g.index.day_of_week / 5)
    f = f.replace([np.inf, -np.inf], np.nan)

    coverage = f.notna().mean()
    keep = coverage[coverage >= cfg.min_feature_coverage].index.tolist()
    if "vol_ewma" not in keep:
        raise RuntimeError("vol_ewma indisponible : historique de l'or insuffisant.")
    f = f[keep]
    # PATCH_DEGEN : variable quasi-constante = artefact fournisseur, pas un fait de marche
    degen = []
    for c_ in f.columns:
        s_ = f[c_].dropna()
        if len(s_) >= 100 and (s_ == 0).mean() > getattr(cfg, "max_zero_fraction", 0.50):
            degen.append(c_)
    if degen:
        print(f"[FEATURES] variables degenerees ecartees : {degen}")
        f = f.drop(columns=degen)
    stale = [c_ for c_ in f.columns if c_ != "vol_ewma" and pd.isna(f[c_].iloc[-1])]
    if stale:
        print(f"[FEATURES] variables sans valeur à la dernière date, écartées : {stale}")
        f = f.drop(columns=stale)
    return f, coverage[f.columns] * 100