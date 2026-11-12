from __future__ import annotations

import math
import shutil
import sys
import textwrap
import unicodedata
from importlib import metadata

import numpy as np
import pandas as pd

try:
    import tabulate as _tab
    _TAB_VER = tuple(int("".join(c for c in p if c.isdigit()) or 0)
                     for p in metadata.version("tabulate").split(".")[:2])
except Exception:
    _tab, _TAB_VER = None, (0, 0)

_UTF = "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()
TABLEFMT = "rounded_grid" if _UTF else "grid"
TEXT_W = min(shutil.get_terminal_size((110, 24)).columns, 110)


# ---------------------------------------------------------------- nombres ---
def fr(value, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", " ")


def pct(v, dec=2, sign=False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{'+' if sign and v > 0 else ''}{fr(v, dec)} %"


def pval(v, threshold=None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return fr(v, 3) + (" *" if threshold is not None and v < threshold else "")


# ----------------------------------------------------------------- tables ---
def _vw(s) -> int:
    return sum(0 if unicodedata.combining(c)
               else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s, width, how):
    gap = max(0, width - _vw(s))
    if how == "right":
        return " " * gap + s
    if how == "center":
        return " " * (gap // 2) + s + " " * (gap - gap // 2)
    return s + " " * gap


def _ascii_table(headers, cells, aligns, wrap):
    n = len(headers)
    aligns = list(aligns) + ["left"] * (n - len(aligns))
    wrap = list(wrap or []) + [None] * (n - len(wrap or []))
    grid = []
    for row in [list(headers)] + [list(r) for r in cells]:
        cols = []
        for i, v in enumerate(row):
            txt = "" if v is None else str(v)
            w = wrap[i] if i < n else None
            cols.append((textwrap.wrap(txt, w) if w else txt.split("\n")) or [""])
        grid.append(cols)
    widths = [max(_vw(l) for r in grid for l in r[i]) for i in range(n)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def render(cols):
        height = max(len(c) for c in cols)
        return "\n".join(
            "|" + "|".join(" " + _pad(cols[i][k] if k < len(cols[i]) else "",
                                      widths[i], aligns[i]) + " " for i in range(n)) + "|"
            for k in range(height))

    out = [sep, render(grid[0]), sep]
    for r in grid[1:]:
        out += [render(r), sep]
    return "\n".join(out)


def md(df, align=None, wrap=None, tablefmt=None) -> str:
    """DataFrame -> table monospace. Ne lève jamais : repli ASCII intégré."""
    n = len(df.columns)
    aligns = tuple(align) if align else ("left",) * n
    if _tab is not None:
        kw = dict(index=False, tablefmt=tablefmt or TABLEFMT,
                  disable_numparse=True, stralign="left")
        if _TAB_VER >= (0, 9):
            kw["colalign"] = aligns
            if wrap:
                kw["maxcolwidths"] = list(wrap)
        try:
            return df.to_markdown(**kw)
        except Exception:
            pass
    cells = [["" if pd.isna(v) else str(v) for v in row]
             for row in df.itertuples(index=False, name=None)]
    return _ascii_table([str(c) for c in df.columns], cells, aligns, wrap)


def para(txt, bullet="  • "):
    print(textwrap.fill(str(txt), width=TEXT_W, initial_indent=bullet,
                        subsequent_indent=" " * len(bullet)))


def title(txt):
    print(f"\n=== {txt} ===")


def safe(fn, *a, **kw):
    """Un échec d'affichage ne doit jamais coûter un run."""
    try:
        return fn(*a, **kw)
    except Exception as exc:
        print(f"[AFFICHAGE] {getattr(fn, '__name__', fn)} a échoué : {exc}")


# --------------------------------------------------------------- sections ---
def print_data(info, cfg):
    title("DONNÉES")
    tab = pd.DataFrame([{
        "Source": cfg.source, "Début": info["start"], "Fin": info["end"],
        "Séances": fr(info["n_sessions"], 0), "Retard": f"{info['staleness_days']} j",
        "Features": str(info["n_features"]),
    }])
    print(md(tab, align=("left", "center", "center", "right", "center", "right")))


def print_quality(q):
    title("QUALITÉ DES DONNÉES")
    rows = []
    for a in q["alerts"]:
        rows.append({"Niveau": "ALERTE", "Date": a["date"], "Constat": a["message"]})
    for m in q["masked"]:
        niveau = "CONSERVÉ" if "conservé" in m["action"] else "MASQUÉ"
        rows.append({"Niveau": niveau, "Date": m["date"],
                     "Constat": f"{m['column']} {pct(m['ret_pct'], 1, True)} — {m['action']}"})
    for r in q["revisions"]:
        rows.append({"Niveau": "RÉVISION", "Date": r["date"],
                     "Constat": f"écart fournisseur {pct(r['diff_pct'], 2)} — historique conservé"})
    if not rows:
        para("Aucune anomalie détectée.", bullet="  ✓ ")
        return
    print(md(pd.DataFrame(rows), align=("center", "center", "left"),
             wrap=[None, None, max(30, TEXT_W - 26)]))
    para("Un mouvement extrême confirmé par un témoin est un fait de marché : "
         "il est conservé, jamais réécrit.", bullet="  ⓘ ")


def print_features(cov_pct, ncols=3):
    title("COUVERTURE DES VARIABLES")
    items = [(c, pct(v, 1)) for c, v in cov_pct.items()]
    nrows = math.ceil(len(items) / ncols)
    cols, aligns, data = [], [], []
    for k in range(ncols):
        chunk = items[k * nrows:(k + 1) * nrows]
        chunk += [("", "")] * (nrows - len(chunk))
        cols += ["Variable", "Couv."]
        aligns += ["left", "right"]
        data += [[a for a, _ in chunk], [b for _, b in chunk]]
    print(md(pd.DataFrame(list(map(list, zip(*data))), columns=cols), align=tuple(aligns)))


def print_selection(table, threshold, n_candidates):
    label = {"naive": "naive (réf.)", "drift": "drift (réf.)"}
    data = {
        "Modèle": [label.get(m, m) for m in table["modele"]],
        "Erreur %": [pct(v * 100) for v in table["MAE_log"]],
        "Erreur $": [fr(v, 1) for v in table["MAE_USD"]],
        "Gain/naive": [pct(v, 2, True) for v in table["gain_vs_naive_%"]],
        "p naive": [pval(v, threshold) for v in table["p_vs_naive"]],
        "Gain/drift": [pct(v, 2, True) for v in table["gain_vs_drift_%"]],
        "p drift": [pval(v, threshold) for v in table["p_vs_drift"]],
        "Direction": [pct(v, 1) for v in table["direction_%"]],
    }
    aligns = ["left", "right", "right", "right", "center", "right", "center", "right"]
    if TEXT_W < 105:
        data.pop("Erreur $")
        aligns.pop(2)
    print(md(pd.DataFrame(data), align=tuple(aligns)))
    para(f"* = p < {fr(threshold, 3)} (α corrigé pour {n_candidates} candidats). "
         "« Erreur % » = MAE en log ×100.", bullet="  ⓘ ")


def print_horizon(res, cfg):
    h = res["horizon"]
    title(f"HORIZON {h} SÉANCE(S) — modèle retenu : {res['model'].upper()}")
    if res["confirmation_note"]:
        para(res["confirmation_note"], bullet="  > ")

    lv = res["levels"]
    tab = pd.DataFrame({
        "Niveau": [f"{k} %" for k in lv],
        "Couverture réelle": [pct(v["coverage"], 1) for v in lv.values()],
        "Demi-largeur": [f"± {fr(v['half_width_pct'], 2)} %" for v in lv.values()],
        "Gain vs fixe": [pct(v["winkler_gain"], 1, True) for v in lv.values()],
        "q base": [fr(v["q_base"], 3) for v in lv.values()],
        "q retenu": [fr(v["q"], 3) for v in lv.values()],
    })
    print(md(tab, align=("center", "center", "right", "right", "right", "right")))
    para(f"{res['calibration_label']}, n = {res['n_calibration']}. Couverture et gain "
         "mesurés sur la moitié tenue à l'écart ; q retenu = conformal adaptatif.",
         bullet="  ⓘ ")

    reg = lv["80"]["regime"]
    if reg:
        title(f"Couverture 80 % par régime de volatilité (h={h})")
        print(md(pd.DataFrame({
            "Régime": [r["label"] for r in reg],
            "n": [str(r["n"]) for r in reg],
            "Couverture": [pct(r["coverage"], 1) for r in reg],
        }), align=("left", "right", "center")))

    print("\nSélection (période ancienne) :")
    print_selection(pd.DataFrame(res["selection_table"]),
                    cfg.alpha_selection / max(res["n_candidates"], 1), res["n_candidates"])

    if not res["validated"]:
        para("Aucun avantage statistiquement prouvé : le centre reste le dernier prix.",
             bullet="  ⓘ ")


def print_reading(rows, cfg):
    if not rows:
        return
    d0, p0 = pd.Timestamp(rows[0]["last_date"]), rows[0]["p0"]
    title(f"LECTURE SIMPLE | {cfg.source.upper()}")
    print(f"Dernière clôture : {d0:%d/%m/%Y}  →  {fr(p0)} $")
    tab = pd.DataFrame({
        "Horizon": [f"{r['horizon']} j" for r in rows],
        "Échéance": [pd.Timestamp(r["target_date"]).strftime("%d/%m") for r in rows],
        "Centre": [f"{fr(r['mid'])} $" for r in rows],
        "Fourchette 80 %": [f"{fr(r['low'])} – {fr(r['high'])}" for r in rows],
        "± 80 %": [f"± {fr((r['high'] - r['low']) / 2 / r['p0'] * 100)} %" for r in rows],
        "Fourchette 90 %": [f"{fr(r['low90'])} – {fr(r['high90'])}" for r in rows],
        "± 90 %": [f"± {fr((r['high90'] - r['low90']) / 2 / r['p0'] * 100)} %" for r in rows],
    })
    print(md(tab, align=("center", "center", "right", "center", "center", "center", "center")))
    print("\nComment lire :")
    for r in rows:
        h = r["horizon"]
        para(f"Dans {h} séance{'s' if h > 1 else ''} "
             f"(≈ {pd.Timestamp(r['target_date']):%d/%m}) : 8 chances sur 10 entre "
             f"{fr(r['low'])} et {fr(r['high'])} $ ; 9 sur 10 entre {fr(r['low90'])} "
             f"et {fr(r['high90'])} $ (centre {fr(r['mid'])} $).")
    para("Rappel : fourchette de probabilité recalée chaque jour sur la dernière clôture, "
         "couverture future non garantie, pas un conseil d'investissement.", bullet="  ! ")


def print_signals(rows, cfg, blocked):
    title(f"DÉCISION | COÛTS {cfg.cost_buffer_pct} % | RISQUE MAX {cfg.risk_pct} %")
    tab = pd.DataFrame({
        "Horizon": [f"{r['horizon']} j" for r in rows],
        "Action": [r["signal"]["action"] for r in rows],
        "P(hausse>coûts)": [pct(100 * r["signal"]["p_up"], 1) for r in rows],
        "P(baisse>coûts)": [pct(100 * r["signal"]["p_dn"], 1) for r in rows],
        "Stop": [f"{fr(r['signal']['stop'])} $" if r["signal"]["stop"] else "—" for r in rows],
        "Notional max": [f"{fr(r['signal']['notional_max'], 0)} $"
                         if r["signal"]["notional_max"] else "—" for r in rows],
    })
    print(md(tab, align=("center", "center", "right", "right", "right", "right")))
    for r in rows:
        para(f"{r['horizon']} j — {r['signal']['reason']}", bullet="  → ")
    if blocked:
        para("COUPE-CIRCUIT ACTIF : la couverture mesurée sur les prévisions échues est "
             "rejetée par le test de Kupiec. Aucune opération tant que la calibration "
             "n'est pas redevenue conforme.", bullet="  ! ")


def print_on_spot(rows, spot):
    title(f"FOURCHETTES RECENTRÉES SUR LE SPOT : {fr(spot)} $")
    tab = pd.DataFrame({
        "Horizon": [f"{r['horizon']} j" for r in rows],
        "Centre": [f"{fr(r['mid'] * spot / r['p0'])} $" for r in rows],
        "80 %": [f"{fr(r['low'] * spot / r['p0'])} – {fr(r['high'] * spot / r['p0'])}" for r in rows],
        "90 %": [f"{fr(r['low90'] * spot / r['p0'])} – {fr(r['high90'] * spot / r['p0'])}" for r in rows],
    })
    print(md(tab, align=("center", "right", "center", "center")))
    para("Le signal et le rapport restent calculés sur la dernière clôture "
         "(le signal est invariant d'échelle).", bullet="  ⓘ ")


def print_pending(tab):
    if tab is None or tab.empty:
        return
    title("MÊME ÉCHÉANCE, INFORMATION ACTUALISÉE")
    print(md(tab, align=("center", "center", "center", "right", "center")))
    para("Une fourchette se déplace avec le marché : ce n'est pas une révision "
         "d'objectif, c'est une nouvelle information.", bullet="  ⓘ ")


def print_tracking(summary, tests):
    if summary is not None and not summary.empty:
        title("PRÉVISIONS ARRIVÉES À ÉCHÉANCE")
        print(md(summary, align=("center", "left", "center", "right", "right", "right", "center")))
    if not tests:
        return
    title("TESTS DE COUVERTURE (prévisions non chevauchantes)")
    tab = pd.DataFrame({
        "Horizon": [f"{t['horizon']} j" for t in tests],
        "Niveau": [f"{t['level']} %" for t in tests],
        "n": [str(t["n"]) for t in tests],
        "Couverture": [pct(t["coverage"], 1) for t in tests],
        "Kupiec p": [pval(t["kupiec_p"], 0.05) for t in tests],
        "Christoffersen p": [pval(t["christoffersen_p"], 0.05) for t in tests],
        "Verdict": [t["verdict"] for t in tests],
    })
    print(md(tab, align=("center", "center", "right", "center", "center", "center", "left")))
    para("* = rejet à 5 %. Kupiec teste le taux de sortie, Christoffersen teste "
         "l'indépendance (les ratés ne doivent pas arriver en rafales).", bullet="  ⓘ ")