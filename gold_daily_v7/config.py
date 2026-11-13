from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_VERSION = "v7.0"

SOURCE_TAGS = {
    "yahoo": "gc_futures",
    "investing":     "xauusd_spot_investing",     # <-- NOUVEAU
    "netdania_csv": "xauusd_spot_netdania",
    "google_csv": "xauusd_spot_google",
    "alphavantage": "xauusd_spot_alphavantage",
}

TICKERS = {
    "dxy": "DX-Y.NYB", "us10y": "^TNX", "vix": "^VIX",
    "oil": "CL=F", "silver": "SI=F", "gld": "GLD",
}


@dataclass
class Config:
    # --- exécution -------------------------------------------------------
    source: str = "yahoo"
    spot: float | None = None
    use_cache: bool = True
    refresh: bool = False
    rebuild: bool = False
    drop_global_holidays: bool = True

    # --- Investing.com ----------------------------------------------------
    investing_id: int = 68             # XAU/USD spot (pair_ID=68)
    investing_resolution: str = "D"
    investing_domain_id: str = "www"
    investing_date_shift: int = 0      # +1 ou -1 si l'audit détecte un décalage
    investing_use_udf: bool = False    # Cloudflare renvoie 403 en permanence

    # --- données ---------------------------------------------------------
    start: str = "2010-01-01"
    gold_ticker: str = "GC=F"
    data_dir: Path = Path("gold_data/v7")
    netdania_csv: Path = Path("data/raw/netdania/XAUUSD.csv")
    google_csv: Path = Path("data/raw/google_finance/XAUUSD.csv")
    alphavantage_csv: Path = Path("data/raw/alphavantage/XAUUSD.csv")
    alphavantage_key: str = field(default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY", ""))
    alphavantage_refresh_hours: int = 20
    csv_dayfirst: bool = True
    netdania_tz: str = "UTC"
    cache_ttl_hours: int = 6
    max_revision_pct: float = 2.0          # % : au-delà, on alerte et on garde l'historique
    ny_close_buffer: tuple[int, int] = (17, 30)
    max_staleness_days: int = 7
    offline: bool = False   # PATCH_SELFTEST_FIX : interdit toute I/O reseau
    max_feature_staleness_rows: int = 2

    # --- qualité (détection, jamais destruction d'un fait de marché) -----
    jump_alert_pct: float = 8.0            # simple, % — alerte seule
    witness_confirm_pct: float = 5.0       # un témoin qui bouge de >5 % confirme le choc
    vix_rel_bound: float = 3.0             # +300 % en une séance
    simple_move_bounds: dict = field(default_factory=lambda: {
        "silver_Close": 0.45, "gld_Close": 0.25, "dxy_Close": 0.12, "oil_Close": 0.40,
    })
    block_on_alert: bool = False

    # --- audit multi-sources --------------------------------------------
    audit_sources: tuple = ("yahoo", "investing", "netdaria_csv", "alphavantage")
    audit_window_days: int = 3 * 365
    divergence_warn_pct: float = 0.75

    # --- features --------------------------------------------------------
    external_lag: int = 1                  # clôtures externes postérieures au settlement or
    external_ffill_limit: int = 3
    min_feature_coverage: float = 0.90
    max_zero_fraction: float = 0.50   # PATCH_DEGEN : >50 % de zeros exacts = ecartee
    drift_window: int = 60

    # --- validation ------------------------------------------------------
    horizons: tuple = (1, 2, 5)
    n_splits: int = 8
    test_size: int = 250
    min_train_rows: int = 500
    selection_fraction: float = 0.50
    confirm_fraction: float = 0.25
    alpha_selection: float = 0.10
    alpha_confirmation: float = 0.10
    n_boot: int = 2000
    boot_block: int = 20
    coverage_levels: tuple = (0.80, 0.90)
    calib_all_if_naive: bool = True
    seed: int = 42

    # --- modèles ---------------------------------------------------------
    ridge_alpha: float = 2000.0
    use_xgb: bool = True
    use_mlp: bool = False                  # -27 % de MAE en V6 : désactivé par défaut
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 250, "learning_rate": 0.02, "max_depth": 2,
        "min_child_weight": 30, "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_lambda": 20, "objective": "reg:squarederror", "tree_method": "hist",
        "random_state": 42, "n_jobs": -1,
    })
    mlp_params: dict = field(default_factory=lambda: {
        "hidden_layer_sizes": (32, 16), "activation": "relu", "alpha": 1e-3,
        "batch_size": 64, "learning_rate_init": 1e-3, "max_iter": 400,
        "early_stopping": True, "validation_fraction": 0.15,
        "n_iter_no_change": 25, "random_state": 42,
    })

    # --- calibration adaptative -----------------------------------------
    aci_gamma: float = 0.05
    aci_window: int = 500
    aci_min_obs: int = 60

    # --- risque / décision ----------------------------------------------
    cost_buffer_pct: float = 0.15
    equity: float = 10_000.0
    risk_pct: float = 0.5                  # % du capital risqué par position
    prob_min: float = 0.60                 # P(gain > coûts) minimale pour agir
    var_alpha: float = 0.05
    var_min_n: int = 100

    # --- chemins ---------------------------------------------------------
    @property
    def raw_dir(self) -> Path: return self.data_dir / "raw"
    @property
    def cache_dir(self) -> Path: return self.data_dir / "cache"
    @property
    def raw_file(self) -> Path: return self.raw_dir / f"raw_prices_{SOURCE_TAGS[self.source]}.csv"
    @property
    def log_file(self) -> Path: return self.data_dir / "forecasts_log_v7.csv"
    @property
    def report_file(self) -> Path: return self.data_dir / "report_v7.json"
    @property
    def oos_file(self) -> Path: return self.data_dir / "oos_predictions_v7.csv"
    @property
    def state_file(self) -> Path: return self.data_dir / "conformal_state_v7.json"

    def mkdirs(self) -> None:
        for p in (self.data_dir, self.raw_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)