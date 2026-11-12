from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import calibration as cal
from . import data, features, models, risk, tracking, ui
from .config import MODEL_VERSION, Config


def final_forecast(raw, feats, h, name, q_by_level, cfg, reg):
    X, y, _ = models.make_dataset(raw, feats, h)
    complete = feats.replace([np.inf, -np.inf], np.nan).dropna()
    if complete.empty:
        raise RuntimeError("Aucune ligne récente avec toutes les features.")
    last = complete.index[-1]
    pos = raw.index.get_indexer([last])[0]
    if pos < 0:
        raise RuntimeError("Dernière date de features absente des prix.")
    stale = len(raw) - 1 - pos
    if stale > cfg.max_feature_staleness_rows:
        raise RuntimeError(f"Features en retard de {stale} séances : mise à jour requise.")
    if stale:
        print(f"[FEATURES] dernière ligne en retard de {stale} séance(s).")

    x_last = complete.loc[[last], X.columns]
    if not np.isfinite(x_last.to_numpy()).all():
        raise RuntimeError("Features non finies pour la prévision finale.")
    mu = models.fit_one(name, X, y, x_last, reg, raw, h, last, cfg)
    scale = float(feats.loc[last, "vol_ewma"] * math.sqrt(h))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Volatilité invalide.")
    p0 = float(raw.loc[last, "gold_Close"])
    mid = p0 * math.exp(mu)

    bands = {}
    for level, q in q_by_level.items():
        lo, hi = p0 * math.exp(mu - scale * q), p0 * math.exp(mu + scale * q)
        bands[str(level)] = {"low": min(lo, mid), "mid": mid, "high": max(hi, mid)}
    return {
        "last_date": last, "target_date": last + pd.offsets.BDay(h), "horizon": h,
        "model": name, "p0": p0, "pred_return_log": mu, "pred_return_pct": mu * 100,
        "scale_log": scale, "vol_daily": scale / math.sqrt(h), "bands": bands,
        "low": bands["80"]["low"], "mid": mid, "high": bands["80"]["high"],
        "low90": bands["90"]["low"], "high90": bands["90"]["high"],
        "version": MODEL_VERSION, "gold_source": cfg.source,
    }


def evaluate_horizon(raw, feats, h, cfg, reg, live_z, state, oos_store):
    y, preds, scale, p0 = models.walk_forward(raw, feats, h, cfg, reg)
    n = len(y)
    n_sel = int(n * cfg.selection_fraction)
    n_conf = int(n * cfg.confirm_fraction)
    if min(n_sel, n_conf, n - n_sel - n_conf) < 100:
        raise RuntimeError(f"OOS insuffisant pour h={h} (n={n}).")

    i_sel = y.index[:n_sel]
    i_conf = y.index[n_sel:n_sel + n_conf]
    i_cal = y.index[n_sel + n_conf:]

    table = models.score_models(y.loc[i_sel], {k: v.loc[i_sel] for k, v in preds.items()},
                                p0.loc[i_sel], cfg, drift=preds["drift"].loc[i_sel])
    n_cand = len(models.candidates(reg))
    best = models.select_model(table, cfg, n_cand)
    note, validated = "", False

    if best != "naive":
        conf = models.score_models(y.loc[i_conf], {best: preds[best].loc[i_conf]},
                                   p0.loc[i_conf], cfg, drift=preds["drift"].loc[i_conf]).iloc[0]
        thr = cfg.alpha_confirmation / 2
        note = (f"confirmation indépendante ({n_conf} obs.) : vs naive gain="
                f"{float(conf['gain_vs_naive_%']):+.2f} %, p={float(conf['p_vs_naive']):.4f} | "
                f"vs drift gain={float(conf['gain_vs_drift_%']):+.2f} %, p={float(conf['p_vs_drift']):.4f}")
        ok = (float(conf["gain_vs_naive_%"]) > 0 and float(conf["p_vs_naive"]) < thr
              and float(conf["gain_vs_drift_%"]) > 0 and float(conf["p_vs_drift"]) < thr)
        if ok:
            validated = True
        else:
            best, note = "naive", note + " -> refusé, retour baseline"

    if best == "naive" and cfg.calib_all_if_naive:
        idx_cal = y.index
        label = "toutes les prévisions hors échantillon"
    else:
        idx_cal = i_cal
        label = "bloc de calibration distinct"

    y_c, mu_c, sc_c = y.loc[idx_cal], preds[best].loc[idx_cal], scale.loc[idx_cal]
    z_c = cal.z_scores(y_c, mu_c, sc_c)

    levels, q_final = {}, {}
    for level in cfg.coverage_levels:
        pct_level = int(round(level * 100))
        alpha = 1 - level
        q_base = cal.conformal_q(z_c, alpha)
        key = f"{h}_{pct_level}"
        aci = (cal.AdaptiveConformal.from_dict(state[key]) if key in state
               else cal.AdaptiveConformal(alpha, cfg.aci_gamma, cfg.aci_window, cfg.aci_min_obs))
        aci.seed(z_c, q_base)
        aci.feed(live_z.get(h, []))
        state[key] = aci.to_dict()
        q = aci.q()
        q_final[pct_level] = q
        levels[str(pct_level)] = {
            "q": q, "q_base": q_base, "alpha_t": aci.alpha_t,
            "coverage": cal.coverage_heldout(y_c, mu_c, sc_c, alpha),
            "winkler_gain": cal.winkler_heldout(y_c, mu_c, sc_c, alpha),
            "half_width_pct": float(q * sc_c.mean() * 100),
            "regime": cal.regime_coverage(y_c, mu_c, sc_c, alpha) if pct_level == 80 else [],
        }

    res = final_forecast(raw, feats, h, best, q_final, cfg, reg)
    res.update({"validated": validated, "selection_table": table.to_dict(orient="records"),
                "confirmation_note": note, "calibration_label": label,
                "n_calibration": int(len(y_c)), "n_candidates": n_cand, "levels": levels})

    oos = pd.DataFrame({"horizon": h, "date": y.index, "y": y.to_numpy(),
                        "p0": p0.to_numpy(), "scale": scale.to_numpy()})
    for k, s in preds.items():
        oos[k] = s.to_numpy()
    oos_store[h] = oos
    return res, z_c


def write_report(cfg, rows, quality, audit, info, feats, tests, summary, state):
    report = {
        "model_version": MODEL_VERSION,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "gold_source": cfg.source, "data": info, "quality": quality, "audit": audit,
        "features": {"n": int(len(feats.columns)), "columns": list(feats.columns)},
        "config": {"horizons": list(cfg.horizons), "coverage_levels": list(cfg.coverage_levels),
                   "external_lag": cfg.external_lag, "cost_buffer_pct": cfg.cost_buffer_pct,
                   "risk_pct": cfg.risk_pct, "prob_min": cfg.prob_min,
                   "aci_gamma": cfg.aci_gamma, "use_mlp": cfg.use_mlp, "use_xgb": cfg.use_xgb,
                   "alpha_selection": cfg.alpha_selection, "baselines": list(models.BASELINES)},
        "var_tests": tests,
        "log_summary": None if summary is None else summary.to_dict(orient="records"),
        "conformal_state": {k: {kk: vv for kk, vv in v.items() if kk != "hist"}
                            for k, v in state.items()},
        "horizons": {str(r["horizon"]): {k: v for k, v in r.items() if k != "selection_table"}
                     | {"selection_table": r["selection_table"]} for r in rows},
    }
    cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[RAPPORT] {cfg.report_file}")


def run(cfg: Config):
    cfg.mkdirs()
    reg = models.registry(cfg)

    raw, revisions = data.load_raw(cfg)
    audit = data.audit_sources(raw, cfg)
    q = data.scan_quality(raw, cfg)
    model, masked, events = data.sanitize_external(raw, cfg)

    quality = {"alerts": q["alerts"], "masked": events, "revisions": revisions,
               "flat_streak": q["flat_streak"], "raw_file": str(cfg.raw_file),
               "note": ("Détection seule. Un choc confirmé par témoin est conservé "
                        "(ex. 30/01/2026). Seules les valeurs non confirmées sont marquées inconnues.")}
    if cfg.block_on_alert and (q["alerts"] or events):
        raise RuntimeError("Anomalies non résolues et block_on_alert=True : arrêt.")

    feats, cov = features.build(model, cfg, masked, q["bad_bars"])
    now_ny = pd.Timestamp.now(tz="America/New_York")
    age = (now_ny.tz_localize(None).normalize() - model.index.max()).days
    if age > cfg.max_staleness_days:
        raise RuntimeError(f"Dernière clôture trop ancienne ({age} jours).")
    info = {"start": str(model.index.min().date()), "end": str(model.index.max().date()),
            "n_sessions": int(len(model)), "staleness_days": int(age),
            "n_features": int(len(feats.columns))}

    # --- échéances AVANT écriture du run courant (exclusion automatique) ---
    journal = tracking.Journal(cfg)
    matured = journal.matured(model)
    live_z = tracking.residuals_by_horizon(matured)
    tests = journal.var_tests(matured)
    summary = journal.summary(matured)
    blocked = risk.circuit_breaker(tests, cfg)

    state = {}
    if cfg.state_file.exists():
        try:
            state = json.loads(cfg.state_file.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    rows, oos_store = [], {}
    for h in cfg.horizons:
        res, z_c = evaluate_horizon(model, feats, h, cfg, reg, live_z, state, oos_store)
        res["signal"] = risk.decide(res, z_c, cfg, blocked)
        rows.append(res)

    if oos_store:
        pd.concat(oos_store.values(), ignore_index=True).to_csv(cfg.oos_file, index=False)
        print(f"[OOS] {cfg.oos_file}")
    cfg.state_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    journal.append(rows)
    pending = journal.pending(rows)

    # --- le rapport est écrit AVANT tout affichage : un crash d'affichage ne coûte rien
    write_report(cfg, rows, quality, audit, info, feats, tests, summary, state)

    ui.safe(ui.print_data, info, cfg)
    ui.safe(ui.print_quality, quality)
    ui.safe(ui.print_features, cov)
    for r in rows:
        ui.safe(ui.print_horizon, r, cfg)
    ui.safe(ui.print_reading, rows, cfg)
    ui.safe(ui.print_signals, rows, cfg, blocked)
    if cfg.spot:
        ui.safe(ui.print_on_spot, rows, cfg.spot)
    ui.safe(ui.print_pending, pending)
    ui.safe(ui.print_tracking, summary, tests)