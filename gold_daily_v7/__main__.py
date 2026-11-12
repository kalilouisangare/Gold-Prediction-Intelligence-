from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from .config import SOURCE_TAGS, Config
from .engine import run


def parse_spot(value):
    if value is None or str(value).lower() in SOURCE_TAGS:
        return None
    s = str(value).strip().replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        print(f"[ARG] spot « {value} » ignoré (attendu : 3975.50 ou 3975,50).")
        return None


def main():
    p = argparse.ArgumentParser(
        prog="gold-v7",
        description="Gold Daily V7 — moteur quantitatif honnête : fourchettes calibrées, "
                    "aucun avantage annoncé sans preuve.",
        epilog="Ne passe aucun ordre. Ne se connecte à aucun courtage.",
    )
    p.add_argument("spot_or_source", nargs="?", default=None,
                   help="prix spot (3975.50 ou 3975,50) OU nom de source")
    p.add_argument("--source", choices=list(SOURCE_TAGS), default=None,
                   help="force la source principale")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore le cache disque")
    p.add_argument("--refresh", action="store_true",
                   help="force le retéléchargement")
    p.add_argument("--rebuild", action="store_true",
                   help="reconstruit tout l'historique (purge les barres figées)")
    p.add_argument("--mlp", action="store_true",
                   help="réactive les candidats réseaux de neurones")
    p.add_argument("--no-xgb", action="store_true",
                   help="retire xgboost et donc le blend")
    p.add_argument("--equity", type=float, default=10_000.0,
                   help="capital de référence pour le sizing (défaut : 10000)")
    p.add_argument("--risk", type=float, default=0.5,
                   help="pourcentage du capital risqué par position (défaut : 0.5)")
    p.add_argument("--prob-min", type=float, default=0.60,
                   help="probabilité minimale de couvrir les coûts (défaut : 0.60)")
    p.add_argument("--data-dir", type=Path, default=Path("gold_data/v7"),
                   help="racine des sorties")
    a = p.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass
    warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    pos_src = str(a.spot_or_source).lower() if a.spot_or_source else None
    cfg = Config(
        source=a.source or (pos_src if pos_src in SOURCE_TAGS else "yahoo"),
        spot=parse_spot(a.spot_or_source),
        use_cache=not a.no_cache, refresh=a.refresh, rebuild=a.rebuild,
        use_mlp=a.mlp, use_xgb=not a.no_xgb,
        equity=a.equity, risk_pct=a.risk, prob_min=a.prob_min, data_dir=a.data_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()