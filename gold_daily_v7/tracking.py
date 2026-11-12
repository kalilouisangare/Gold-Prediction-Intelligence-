from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from .calibration import christoffersen, kupiec
from .config import MODEL_VERSION, Config
from . import ui


def _flat(r: dict) -> dict:
    lv = r["levels"]
    return {
        "anchor_date": pd.Timestamp(r["last_date"]).date(),
        "target_date": pd.Timestamp(r["target_date"]).date(),
        "horizon": r["horizon"], "model": r["model"],
        "validated": bool(r["validated"]), "signal": r["signal"]["action"],
        "p0": r["p0"], "pred_return_log": r["pred_return_log"], "scale_log": r["scale_log"],
        "low": r["low"], "mid": r["mid"], "high": r["high"],
        "low90": r["low90"], "high90": r["high90"],
        "q80": lv["80"]["q"], "q90": lv["90"]["q"],
        "coverage_80_heldout": lv["80"]["coverage"], "coverage_90_heldout": lv["90"]["coverage"],
        "version": r["version"], "gold_source": r["gold_source"],
    }


class Journal:
    KEY = ["anchor_date", "horizon", "version", "gold_source"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.log_file

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.path)
        if df.empty:
            return df
        for c in ("anchor_date", "target_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
        df = df.loc[df["gold_source"] == self.cfg.source]
        return df.sort_values("emis_le").drop_duplicates(self.KEY, keep="first")

    def append(self, rows) -> str:
        run_id = str(uuid.uuid4())
        new = pd.DataFrame([_flat(r) for r in rows])
        new["run_id"] = run_id
        new["forecast_id"] = [str(uuid.uuid4()) for _ in range(len(new))]
        new["emis_le"] = pd.Timestamp.now(tz="UTC").isoformat()
        old = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()
        combined = pd.concat([old, new], ignore_index=True, sort=False)
        combined["anchor_date"] = pd.to_datetime(combined["anchor_date"], errors="coerce").dt.normalize()
        combined = (combined.sort_values("emis_le")
                    .drop_duplicates(self.KEY, keep="first"))
        combined.to_csv(self.path, index=False)
        print(f"[LOG] {self.path} | run_id={run_id}")
        return run_id

    # ------------------------------------------------ échéances réalisées ---
    def matured(self, raw: pd.DataFrame) -> pd.DataFrame:
        log = self.read()
        if log.empty:
            return pd.DataFrame()
        pos = pd.Series(np.arange(len(raw)), index=raw.index.normalize())
        close = raw["gold_Close"].astype(float)
        out = []
        for _, r in log.iterrows():
            a = pos.get(r["anchor_date"])
            if a is None:
                continue
            h = int(r["horizon"])
            t = int(a) + h
            if t >= len(raw):
                continue
            actual = float(close.iloc[t])
            p0 = float(r["p0"])
            y = float(np.log(actual / p0))
            sc = float(r["scale_log"])
            out.append({
                "horizon": h, "model": r["model"], "version": r["version"],
                "anchor": r["anchor_date"], "target": raw.index[t], "anchor_pos": int(a),
                "p0": p0, "actual": actual, "mid": float(r["mid"]),
                "y": y, "z": (y - float(r["pred_return_log"])) / sc if sc > 0 else np.nan,
                "err_model": abs(actual - float(r["mid"])), "err_naive": abs(actual - p0),
                "in80": float(r["low"]) <= actual <= float(r["high"]),
                "in90": float(r["low90"]) <= actual <= float(r["high90"]),
            })
        return pd.DataFrame(out)

    @staticmethod
    def summary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        g = (df.groupby(["horizon", "model", "version"])
             .agg(n=("err_model", "size"), MAE_model=("err_model", "mean"),
                  MAE_naive=("err_naive", "mean"), couverture=("in80", "mean")).reset_index())
        return pd.DataFrame({
            "Horizon": [f"{int(v)} j" for v in g["horizon"]],
            "Modèle": g["model"].astype(str), "Version": g["version"].astype(str),
            "n": [str(int(v)) for v in g["n"]],
            "MAE modèle": [f"{ui.fr(v, 1)} $" for v in g["MAE_model"]],
            "MAE naive": [f"{ui.fr(v, 1)} $" for v in g["MAE_naive"]],
            "Couv. 80 %": [ui.pct(100 * v, 1) for v in g["couverture"]],
        })

    @staticmethod
    def non_overlapping(df: pd.DataFrame, h: int) -> pd.DataFrame:
        """Une prévision tous les h jours d'ancrage : sinon les tests sont biaisés."""
        d = df.loc[df["horizon"] == h].sort_values("anchor_pos")
        keep, last = [], -10**9
        for i, r in d.iterrows():
            if int(r["anchor_pos"]) - last >= h:
                keep.append(i)
                last = int(r["anchor_pos"])
        return d.loc[keep]

    def var_tests(self, df: pd.DataFrame) -> list[dict]:
        if df.empty:
            return []
        out = []
        for h in sorted(df["horizon"].unique()):
            d = self.non_overlapping(df, int(h))
            for level, col in ((80, "in80"), (90, "in90")):
                n = len(d)
                if n < 20:
                    continue
                hits = (~d[col].astype(bool)).to_numpy().astype(int)
                x = int(hits.sum())
                _, pk = kupiec(n, x, 1 - level / 100)
                _, pc = christoffersen(hits)
                cov = 100 * (1 - x / n)
                verdict = "conforme"
                if np.isfinite(pk) and pk < self.cfg.var_alpha and cov < level:
                    verdict = "trop étroit"
                elif np.isfinite(pc) and pc < self.cfg.var_alpha:
                    verdict = "ratés groupés"
                out.append({"horizon": int(h), "level": level, "n": n, "misses": x,
                            "coverage": cov, "kupiec_p": pk,
                            "christoffersen_p": pc, "verdict": verdict})
        return out

    # --------------------------------------------------- réconciliation ---
    def pending(self, rows) -> pd.DataFrame:
        log = self.read()
        if log.empty:
            return pd.DataFrame()
        targets = [pd.Timestamp(r["target_date"]) for r in rows]
        d = log.loc[log["target_date"].isin(targets)].sort_values(["target_date", "emis_le"])
        if d.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "Cible": [f"{t:%d/%m}" for t in d["target_date"]],
            "Émise le": [f"{t:%d/%m}" for t in d["anchor_date"]],
            "h": [str(int(v)) for v in d["horizon"]],
            "Ancre": [f"{ui.fr(v)} $" for v in d["p0"]],
            "80 %": [f"{ui.fr(a)} – {ui.fr(b)}" for a, b in zip(d["low"], d["high"])],
        })


def residuals_by_horizon(matured: pd.DataFrame) -> dict:
    """(date_cible_iso, z) triés, pour alimenter le conformal adaptatif."""
    if matured.empty:
        return {}
    out = {}
    for h, g in matured.groupby("horizon"):
        g = g.sort_values("target")
        out[int(h)] = [(f"{t:%Y-%m-%d}", float(z)) for t, z in zip(g["target"], g["z"])]
    return out