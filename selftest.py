"""
Test hors-ligne de V7 : aucun appel réseau.
Déroule engine.run deux fois, puis vérifie les correctifs clés.
    python selftest.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gold_daily_v7 import data, engine
from gold_daily_v7.calibration import AdaptiveConformal, christoffersen, kupiec
from gold_daily_v7.config import Config

OK, KO = "  [PASS]", "  [FAIL]"
fails = []


def check(label, cond):
    print((OK if cond else KO), label)
    if not cond:
        fails.append(label)


def synthetic(n=3400, seed=42):
    rng = np.random.default_rng(seed)
    # PATCH_SELFTEST_FIX : fin ancree sur aujourd hui. Sinon la serie figee
    # vieillit et le garde-fou max_staleness_days la rejette.
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    # volatilité en clusters + queues épaisses
    r = np.zeros(n)
    v = 0.010 ** 2
    for t in range(n):
        v = 2e-6 + 0.09 * (r[t - 1] ** 2 if t else 0.0) + 0.89 * v
        r[t] = np.sqrt(v) * rng.standard_t(5) / np.sqrt(5 / 3)
    close = 1800.0 * np.exp(np.cumsum(r))

    df = pd.DataFrame(index=idx)
    df["gold_Close"] = close
    df["gold_Open"] = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, .002, n))
    df["gold_High"] = close * (1 + np.abs(rng.normal(0, .005, n)))
    df["gold_Low"] = close * (1 - np.abs(rng.normal(0, .005, n)))
    df["gold_Volume"] = rng.integers(80_000, 250_000, n).astype(float)

    for name, base, sd in [("dxy", 100., .004), ("us10y", 4.0, .015),
                           ("vix", 18., .06), ("oil", 70., .018), ("silver", 24., .018)]:
        df[f"{name}_Close"] = base * np.exp(np.cumsum(rng.normal(0, sd, n)))
    df["gld_Close"] = df["gold_Close"] / 10 * (1 + rng.normal(0, .001, n))

    events = {}
    # 1) krach RÉEL : or -11.4 %, argent -31.4 %, GLD suit -> doit être CONSERVÉ
    i = 2000
    df.iloc[i, df.columns.get_loc("gold_Close")] *= 0.886
    df.iloc[i, df.columns.get_loc("silver_Close")] *= 0.686
    df["gld_Close"] = df["gold_Close"] / 10 * (1 + rng.normal(0, .001, n))
    events["crash"] = idx[i]
    # 2) spike argent ISOLÉ, aucun témoin -> doit être MASQUÉ
    j = 2500
    df.iloc[j, df.columns.get_loc("silver_Close")] *= 0.50
    events["spike"] = idx[j]
    # 3) barre OHLC incohérente -> range_hl/close_pos masqués
    k = 2600
    df.iloc[k, df.columns.get_loc("gold_High")] = df["gold_Low"].iloc[k] * 0.99
    events["badbar"] = idx[k]
    return df, events


def main():
    full, ev = synthetic()
    tmp = Path(tempfile.mkdtemp(prefix="gold_v7_selftest_"))
    print(f"Sorties temporaires : {tmp}\n")

    # PATCH_SELFTEST_FIX : offline=True (aucun reseau) ; RUN 1 tronque 60
    # seances (~84 j calendaires) -> fraicheur relaxee pour le test seulement.
    common = dict(rebuild=True, use_xgb=False, use_mlp=False, n_boot=200,
                  data_dir=tmp, offline=True, max_staleness_days=120)

    # ---- run 1 : historique tronqué de 60 séances (les prévisions vont mûrir) ----
    data.fetch_all = lambda cfg, start, cache: full.iloc[:-60].copy()
    cfg1 = Config(**common)
    print("========== RUN 1 ==========")
    engine.run(cfg1)

    # ---- run 2 : historique complet, même dossier -> journal + échéances + état ----
    data.fetch_all = lambda cfg, start, cache: full.copy()
    cfg2 = Config(**{**common, "rebuild": False})
    print("\n========== RUN 2 ==========")
    engine.run(cfg2)

    # ---- vérifications ciblées ----
    print("\n========== VÉRIFICATIONS ==========")
    cfg = Config(data_dir=tmp)
    raw, _ = data.load_raw(cfg)
    model, masked, events = data.sanitize_external(raw, cfg)
    # PATCH_SELFTEST_ASSERTS : le choc synthetique (-31.4 %) est sous la borne
    # silver_Close (45 %) ; seul le rebond J+1 (+45.8 %) la franchit et est trace.
    _cp = full.index.get_loc(ev["crash"])
    crash_days = {f"{full.index[_cp]:%Y-%m-%d}",
                  f"{full.index[min(_cp + 1, len(full) - 1)]:%Y-%m-%d}"}

    check("krach réel conservé (argent non effacé)",
          pd.notna(model.loc[ev["crash"], "silver_Close"]))
    check("spike isolé masqué (argent NaN)",
          pd.isna(model.loc[ev["spike"], "silver_Close"]))
    check("le krach est bien tracé comme confirmé",
          any(e["date"] in crash_days and "conservé" in e["action"] for e in events))
    check("barre OHLC incohérente détectée",
          ev["badbar"] in data.scan_quality(raw, cfg)["bad_bars"])

    from gold_daily_v7 import features
    feats, cov = features.build(model, cfg, masked, data.scan_quality(raw, cfg)["bad_bars"])
    check("range_hl masqué sur la barre incohérente",
          pd.isna(feats.loc[ev["badbar"], "range_hl"]) if "range_hl" in feats else True)
    check("vol_ewma présente et finie", bool(np.isfinite(feats["vol_ewma"].iloc[1:]).all()))

    # Kupiec : 250 tirages, 50 sorties à 80 % -> conforme ; 75 -> rejet
    _, p_ok = kupiec(250, 50, 0.20)
    _, p_ko = kupiec(250, 75, 0.20)
    check(f"Kupiec conforme (p={p_ok:.3f} > .05)", p_ok > 0.05)
    check(f"Kupiec rejet (p={p_ko:.4f} < .05)", p_ko < 0.05)

    # Christoffersen : sorties alternées vs groupées
    # PATCH_SELFTEST_ASSERTS : 0,1,0,1... est maximalement dependante
    # (n00 = n11 = 0) : le test d independance DOIT rejeter. La non-rejection
    # se teste sur une sequence bernoullienne independante.
    alt = (np.random.default_rng(7).random(400) < 0.20).astype(int)
    grp = np.r_[np.zeros(150), np.ones(150)]
    _, p_alt = christoffersen(alt)
    _, p_grp = christoffersen(grp)
    check(f"Christoffersen independant non rejete (p={p_alt:.3f})", p_alt > 0.05)
    check(f"Christoffersen groupé rejeté (p={p_grp:.2e})", p_grp < 0.05)

    # Conformal adaptatif : trop de ratés -> q AUGMENTE ; trop peu -> q DIMINUE
    # PATCH_ACI_DATES : feed() exige des dates strictement croissantes.
    # '1 + _ % 28' ne fournit que 28 cles valides < min_obs (60) : q() restait
    # fige sur base_q. Dates distinctes + 120 iterations -> warmup franchi.
    z = list(np.abs(np.random.default_rng(0).standard_t(5, 400) / np.sqrt(5 / 3)))
    a = AdaptiveConformal(0.20, 0.05, 500, 60); a.seed(z, 1.3); q0 = a.q()
    for _ in range(120):
        a.feed([((pd.Timestamp("2099-01-01") + pd.Timedelta(days=_)).strftime("%Y-%m-%d"), 9.0)])       # ratés systématiques
    check(f"adaptatif : ratés -> q élargit ({q0:.3f} -> {a.q():.3f})", a.q() > q0)
    b = AdaptiveConformal(0.20, 0.05, 500, 60); b.seed(z, 1.3); q0b = b.q()
    for _ in range(120):
        b.feed([((pd.Timestamp("2099-01-01") + pd.Timedelta(days=_)).strftime("%Y-%m-%d"), 0.0)])       # jamais de raté
    check(f"adaptatif : zéro raté -> q rétrécit ({q0b:.3f} -> {b.q():.3f})", b.q() < q0b)

    check("rapport JSON écrit", cfg.report_file.exists())
    check("journal écrit", cfg.log_file.exists())
    check("état conformal écrit", cfg.state_file.exists())

    print("\n" + ("SELFTEST OK — tu peux lancer le vrai run."
                  if not fails else f"SELFTEST : {len(fails)} échec(s) -> {fails}"))


if __name__ == "__main__":
    main()