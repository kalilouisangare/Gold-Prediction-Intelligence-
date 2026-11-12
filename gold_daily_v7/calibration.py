from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def z_scores(y, mu, scale) -> np.ndarray:
    z = (np.asarray(y, float) - np.asarray(mu, float)) / np.maximum(np.asarray(scale, float), 1e-8)
    return z[np.isfinite(z)]


def conformal_q(z, alpha: float) -> float:
    z = np.abs(np.asarray(z, float))
    z = z[np.isfinite(z)]
    if len(z) < 50:
        raise RuntimeError("Pas assez de résidus pour calibrer.")
    return float(np.quantile(z, 1 - alpha))


def heldout_split(y, mu, scale):
    half = len(y) // 2
    if half < 30 or len(y) - half < 30:
        return None
    return (y.iloc[:half], mu.iloc[:half], scale.iloc[:half],
            y.iloc[half:], mu.iloc[half:], scale.iloc[half:])


def coverage_heldout(y, mu, scale, alpha) -> float:
    p = heldout_split(y, mu, scale)
    if p is None:
        return float("nan")
    yf, mf, sf, yt, mt, st = p
    q = conformal_q(z_scores(yf, mf, sf), alpha)
    return 100 * float(np.mean(np.abs(z_scores(yt, mt, st)) <= q))


def winkler_heldout(y, mu, scale, alpha) -> float:
    p = heldout_split(y, mu, scale)
    if p is None:
        return float("nan")
    yf, mf, sf, yt, mt, st = p
    q = conformal_q(z_scores(yf, mf, sf), alpha)
    const = float(np.quantile(np.abs(yf.to_numpy() - mf.to_numpy()), 1 - alpha))
    yv, mv, sv = yt.to_numpy(float), mt.to_numpy(float), st.to_numpy(float)

    def score(lo, hi):
        return float(np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - yv, 0)
                             + (2 / alpha) * np.maximum(yv - hi, 0)))

    s_var = score(mv - q * sv, mv + q * sv)
    s_fix = score(mv - const, mv + const)
    return 100 * (1 - s_var / s_fix) if s_fix > 0 else float("nan")


def regime_coverage(y, mu, scale, alpha, n_bins: int = 3) -> list[dict]:
    """Couverture tenue à l'écart, conditionnée au régime de volatilité connu à t."""
    p = heldout_split(y, mu, scale)
    if p is None:
        return []
    yf, mf, sf, yt, mt, st = p
    q = conformal_q(z_scores(yf, mf, sf), alpha)
    edges = np.quantile(sf.to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1])
    buckets = np.digitize(st.to_numpy(), edges)
    inside = np.abs(z_scores(yt, mt, st)) <= q
    labels = ["vol basse", "vol moyenne", "vol haute"][:n_bins]
    out = []
    for b in range(n_bins):
        m = buckets == b
        if m.sum() >= 20:
            out.append({"label": labels[b], "n": int(m.sum()),
                        "coverage": 100 * float(inside[m].mean())})
    return out


# ----------------------------------------------------- conformal adaptatif ---
@dataclass
class AdaptiveConformal:
    """Gibbs & Candès : on met à jour le NIVEAU alpha, jamais q directement."""
    alpha: float
    gamma: float = 0.05
    window: int = 500
    min_obs: int = 60
    base_q: float = 1.28
    alpha_t: float | None = None
    hist: list = field(default_factory=list)
    last_fed: str | None = None
    n_live: int = 0

    def __post_init__(self):
        if self.alpha_t is None:
            self.alpha_t = self.alpha

    def seed(self, z, base_q: float):
        self.base_q = float(base_q)
        if not self.hist:
            self.hist = [float(abs(v)) for v in np.asarray(z, float) if np.isfinite(v)][-self.window:]

    def q(self) -> float:
        if self.n_live < self.min_obs:
            return self.base_q
        lvl = min(max(1.0 - self.alpha_t, 0.50), 0.999)
        return float(np.quantile(self.hist, lvl))

    def feed(self, observations):
        """observations : liste triée (date_iso, z_realise). Jamais deux fois la même."""
        for date_iso, z in observations:
            if self.last_fed is not None and date_iso <= self.last_fed:
                continue
            if not np.isfinite(z):
                continue
            err = int(abs(z) > self.q())
            self.alpha_t = float(min(max(self.alpha_t + self.gamma * (self.alpha - err), 0.005), 0.60))
            self.hist.append(float(abs(z)))
            self.hist = self.hist[-self.window:]
            self.last_fed = date_iso
            self.n_live += 1

    def to_dict(self):
        return {"alpha": self.alpha, "gamma": self.gamma, "window": self.window,
                "min_obs": self.min_obs, "base_q": self.base_q, "alpha_t": self.alpha_t,
                "hist": self.hist, "last_fed": self.last_fed, "n_live": self.n_live}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# --------------------------------------------------------- tests de VaR ---
def _chi2_sf1(x: float) -> float:
    """P(chi2_1 > x) = erfc(sqrt(x/2)) — exact, sans scipy."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2))


def kupiec(n: int, x: int, p: float):
    """H0 : le taux de sortie vaut p. x = sorties observées sur n."""
    if n == 0:
        return float("nan"), float("nan")
    pi = x / n
    if x == 0:
        ll0 = ll1 = n * math.log(1 - p)
    elif x == n:
        ll0 = ll1 = n * math.log(p)
    else:
        ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
        ll1 = (n - x) * math.log(1 - pi) + x * math.log(pi)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, _chi2_sf1(lr)


def christoffersen(hits):
    """H0 : les sorties sont indépendantes (pas de rafales)."""
    h = np.asarray(hits, dtype=int)
    if len(h) < 20:
        return float("nan"), float("nan")
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(len(h) - 1, 1)

    def ll(pi, a, b):
        return a * math.log(1 - pi) + b * math.log(pi) if 0 < pi < 1 else 0.0

    lr = max(-2 * (ll(p, n00, n01) + ll(p, n10, n11)
                   - ll(p01, n00, n01) - ll(p11, n10, n11)), 0.0)
    return lr, _chi2_sf1(lr)