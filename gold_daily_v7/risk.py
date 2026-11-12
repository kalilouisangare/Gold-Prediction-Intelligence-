from __future__ import annotations

import math

import numpy as np

from .config import Config


def probabilities(mu: float, scale: float, cost_pct: float, z_emp) -> tuple[float, float]:
    """P(rendement > coûts) et P(rendement < -coûts), loi empirique des résidus."""
    z = np.asarray(z_emp, float)
    z = z[np.isfinite(z)]
    if len(z) < 50 or scale <= 0:
        return float("nan"), float("nan")
    c = math.log1p(cost_pct / 100.0)
    return float((z > (c - mu) / scale).mean()), float((z < (-c - mu) / scale).mean())


def decide(res: dict, z_emp, cfg: Config, blocked: bool = False) -> dict:
    p0 = float(res["p0"])
    mu = float(res["pred_return_log"])
    scale = float(res["scale_log"])
    p_up, p_dn = probabilities(mu, scale, cfg.cost_buffer_pct, z_emp)

    base = {"p_up": p_up, "p_dn": p_dn, "stop": None, "notional_max": None,
            "cost_pct": cfg.cost_buffer_pct, "risk_pct": cfg.risk_pct,
            "prob_min": cfg.prob_min, "validated": bool(res.get("validated"))}

    if blocked:
        return {**base, "action": "NEUTRAL",
                "reason": "Coupe-circuit : couverture rejetée par Kupiec sur les prévisions échues."}
    if not res.get("validated"):
        return {**base, "action": "NEUTRAL",
                "reason": "Aucun modèle ne bat naive ET drift avec preuve statistique."}
    if not np.isfinite(p_up) or max(p_up, p_dn) < cfg.prob_min:
        return {**base, "action": "NEUTRAL",
                "reason": f"Probabilité de couvrir les coûts < {cfg.prob_min:.0%}."}

    buy = p_up > p_dn
    stop = float(res["low90"]) if buy else float(res["high90"])
    dist = abs(p0 - stop) / p0
    if dist <= 0:
        return {**base, "action": "NEUTRAL", "reason": "Distance au stop nulle : sizing impossible."}
    notional = cfg.equity * (cfg.risk_pct / 100.0) / dist
    return {**base, "action": "ACHAT" if buy else "VENTE", "stop": stop,
            "notional_max": notional,
            "reason": (f"P({'hausse' if buy else 'baisse'} > coûts) = {max(p_up, p_dn):.0%} ; "
                       f"stop sur la borne 90 % à {dist * 100:.2f} % ; "
                       f"risque plafonné à {cfg.risk_pct} % du capital.")}


def circuit_breaker(tests: list[dict], cfg: Config) -> bool:
    """Blocage si la couverture mesurée est significativement TROP BASSE."""
    for t in tests:
        if (t["n"] >= cfg.var_min_n and np.isfinite(t["kupiec_p"])
                and t["kupiec_p"] < cfg.var_alpha and t["coverage"] < t["level"]):
            return True
    return False