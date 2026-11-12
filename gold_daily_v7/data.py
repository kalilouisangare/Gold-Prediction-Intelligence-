from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import SOURCE_TAGS, TICKERS, Config


# ------------------------------------------------------------- index / OHLC ---
def normalise_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    valid = ~idx.isna()
    out, idx = out.loc[valid].copy(), idx[valid]
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _flatten_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        chosen = next((l for l in range(out.columns.nlevels)
                       if {str(v) for v in out.columns.get_level_values(l)} & wanted), None)
        if chosen is None:
            raise ValueError("Format de colonnes Yahoo non reconnu.")
        out.columns = out.columns.get_level_values(chosen)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_log_ret(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    return np.log(v.where(v > 0)).diff()


def simple_ret(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    return v.where(v > 0).pct_change()


def last_completed_market_day(now_ny: pd.Timestamp, cfg: Config) -> pd.Timestamp:
    h, m = cfg.ny_close_buffer
    return (now_ny - pd.Timedelta(hours=h, minutes=m)).tz_localize(None).normalize()


# ------------------------------------------------------------------- cache ---
class YahooCache:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def path(self, ticker: str) -> Path:
        return self.cfg.cache_dir / f"{ticker.replace('=', '_').replace('^', '')}.csv"

    def read(self, ticker: str, start: str) -> pd.DataFrame:
        p = self.path(ticker)
        if not p.exists():
            return pd.DataFrame()
        try:
            d = normalise_daily_index(pd.read_csv(p, index_col=0, parse_dates=True))
        except Exception:
            return pd.DataFrame()
        if d.empty:
            return d
        d = _numeric(d)
        return d.loc[d.index >= pd.Timestamp(start)].sort_index()

    def write(self, ticker: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        try:
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            old = self.read(ticker, "1900-01-01")
            merged = df if old.empty else df.combine_first(old).sort_index()
            merged.to_csv(self.path(ticker))
        except Exception as exc:
            print(f"[CACHE] {ticker}: écriture impossible ({exc}).")

    def fresh(self, ticker: str) -> bool:
        p = self.path(ticker)
        if not p.exists():
            return False
        age = pd.Timestamp.now(tz="UTC") - pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)
        return age < pd.Timedelta(hours=self.cfg.cache_ttl_hours)

    def download(self, ticker: str, start: str, prefer_cache: bool = True) -> pd.DataFrame:
        cfg = self.cfg
        if cfg.use_cache and prefer_cache and not cfg.refresh and self.fresh(ticker):
            d = self.read(ticker, start)
            if not d.empty and "Close" in d.columns:
                return d
        try:
            kw = dict(start=start, interval="1d", auto_adjust=False,
                      progress=False, threads=False, timeout=20)
            try:
                d = yf.download(ticker, multi_level_index=False, **kw)
            except TypeError:
                d = yf.download(ticker, **kw)
        except Exception as exc:
            print(f"[Yahoo] {ticker}: {exc}")
            d = pd.DataFrame()
        if d is not None and not d.empty:
            try:
                d = _numeric(normalise_daily_index(_flatten_yahoo(d)))
            except ValueError as exc:
                print(f"[Yahoo] {ticker}: {exc}")
                d = pd.DataFrame()
        if d is not None and not d.empty and "Close" in d.columns:
            if cfg.use_cache:
                self.write(ticker, d)
            return d
        if cfg.use_cache:
            d = self.read(ticker, start)
            if not d.empty and "Close" in d.columns:
                print(f"[Yahoo] {ticker}: téléchargement échoué, repli sur le cache.")
                return d
        return pd.DataFrame()


# --------------------------------------------------------------- CSV local ---
def parse_dates(values: pd.Series, dayfirst: bool) -> pd.DatetimeIndex:
    s = values.astype(str).str.strip()
    if len(s) == 0:
        return pd.DatetimeIndex([])
    if s.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
        try:
            return pd.DatetimeIndex(pd.to_datetime(s, errors="coerce", format="ISO8601"))
        except (ValueError, TypeError):
            return pd.DatetimeIndex(pd.to_datetime(s, errors="coerce", utc=True))
    return pd.DatetimeIndex(pd.to_datetime(s, errors="coerce", dayfirst=dayfirst))


def intraday_to_ny_daily(df: pd.DataFrame, tz: str, cutoff_hour: int = 17) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    work = df.copy()
    work.index = idx
    work = work.loc[~work.index.isna()].sort_index()
    for c in ("Open", "High", "Low"):
        if c not in work.columns:
            work[c] = work["Close"]
    ny = work.index.tz_convert("America/New_York")
    day = (ny + pd.Timedelta(hours=24 - cutoff_hour)).tz_localize(None).normalize()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in work.columns:
        agg["Volume"] = "sum"
    out = work.groupby(day).agg(agg)
    return out[out.index.day_of_week < 5]


def read_local_csv(path: Path, label: str, cfg: Config,
                   allow_intraday: bool = False, tz: str = "UTC",
                   verbose: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier {label} absent : {path}.")
    try:
        d = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise RuntimeError(f"Lecture impossible du CSV {label} : {exc}") from exc

    key = lambda v: "".join(c.lower() for c in str(v).strip() if c.isalnum())
    aliases = {
        "date": {"date", "datetime", "timestamp", "time", "dateheure", "jour"},
        "open": {"open", "opening", "ouverture"},
        "high": {"high", "highest", "haut", "plushaut"},
        "low": {"low", "lowest", "bas", "plusbas"},
        "close": {"close", "last", "price", "cours", "cloture"},
        "volume": {"volume", "vol"},
    }
    found: dict[str, str] = {}
    for col in d.columns:
        k = key(col)
        for target, names in aliases.items():
            if k in names and target not in found:
                found[target] = col
    if "date" not in found or "close" not in found:
        raise ValueError(f"{label} : colonnes Date et Close requises, vu {list(d.columns)}.")

    out = pd.DataFrame(index=parse_dates(d[found["date"]], cfg.csv_dayfirst))
    out = out.loc[out.index.notna()]

    def parse_number(v: str) -> str:
        if "," in v and "." in v:
            return (v.replace(".", "").replace(",", ".")
                    if v.rfind(",") > v.rfind(".") else v.replace(",", ""))
        return v.replace(",", ".") if "," in v else v

    for target in ("open", "high", "low", "close", "volume"):
        if target not in found:
            continue
        v = d[found[target]]
        if v.dtype == object:
            v = (v.astype(str).str.strip()
                 .str.replace(" ", "", regex=False)
                 .str.replace("\u00a0", "", regex=False).map(parse_number))
        out[target] = pd.to_numeric(v, errors="coerce").to_numpy()

    out = out.rename(columns={c: c.capitalize() for c in out.columns})
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    if out.empty:
        raise ValueError(f"{label} : aucune clôture numérique exploitable.")

    if allow_intraday:
        idx = pd.DatetimeIndex(out.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        if pd.Series(idx.normalize()).duplicated().any():
            if verbose:
                print(f"[{label.upper()}] intrajournalier -> journées à la coupure 17h NY.")
            return intraday_to_ny_daily(out, tz=tz)
    daily = normalise_daily_index(out)
    return daily[daily.index.day_of_week < 5].sort_index()


# ----------------------------------------------------------- Alpha Vantage ---
def fetch_alphavantage(cfg: Config, start: str) -> pd.DataFrame:
    path = cfg.alphavantage_csv
    fresh = (path.exists() and
             pd.Timestamp.now(tz="UTC") - pd.to_datetime(path.stat().st_mtime, unit="s", utc=True)
             < pd.Timedelta(hours=cfg.alphavantage_refresh_hours))
    if not fresh and cfg.alphavantage_key:
        try:
            import requests
            r = requests.get("https://www.alphavantage.co/query", timeout=30, params={
                "function": "GOLD_SILVER_HISTORY", "symbol": "GOLD",
                "interval": "daily", "apikey": cfg.alphavantage_key})
            r.raise_for_status()
            payload = r.json()
            rows = [{"Date": i["date"], "Close": i["price"]}
                    for i in payload.get("data", [])
                    if isinstance(i, dict) and i.get("date") and i.get("price")]
            if rows:
                df = pd.DataFrame(rows)
                df.index = parse_dates(df["Date"], cfg.csv_dayfirst)
                df = normalise_daily_index(df.drop(columns=["Date"]))
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                df = df[["Close"]].dropna()
                df = df[df["Close"] > 0]
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index_label="Date")
            else:
                print("[ALPHA VANTAGE] aucune série reçue.")
        except Exception as exc:
            print(f"[ALPHA VANTAGE] {exc}")
    if not path.exists():
        return pd.DataFrame()
    df = read_local_csv(path, "Alpha Vantage", cfg, verbose=False)
    return df.loc[df.index >= pd.Timestamp(start)].sort_index()


# --------------------------------------------------------------- collecte ---
def fetch_gold(cfg: Config, start: str, cache: YahooCache) -> pd.DataFrame:
    if cfg.source == "yahoo":
        return cache.download(cfg.gold_ticker, start, prefer_cache=False)
    if cfg.source == "netdania_csv":
        g = read_local_csv(cfg.netdania_csv, "NetDania", cfg, True, cfg.netdania_tz)
    elif cfg.source == "google_csv":
        g = read_local_csv(cfg.google_csv, "Google Finance", cfg)
    elif cfg.source == "alphavantage":
        return fetch_alphavantage(cfg, start)
    else:
        raise ValueError(f"Source inconnue : {cfg.source}")
    return g.loc[g.index >= pd.Timestamp(start)]


def fetch_all(cfg: Config, start: str, cache: YahooCache) -> pd.DataFrame:
    parts = []
    gold = fetch_gold(cfg, start, cache)
    if gold.empty or "Close" not in gold.columns:
        print("[OR] aucune clôture pour la source principale.")
    else:
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in gold.columns]
        parts.append(gold[cols].rename(columns={c: f"gold_{c}" for c in cols}))
    for name, ticker in TICKERS.items():
        d = cache.download(ticker, start)
        if d.empty or "Close" not in d.columns:
            print(f"[EXTERNE] {name} ({ticker}) indisponible.")
            continue
        parts.append(d[["Close"]].rename(columns={"Close": f"{name}_Close"}))
    return pd.concat(parts, axis=1, sort=False).sort_index() if parts else pd.DataFrame()


def merge_guard(new: pd.DataFrame, old: pd.DataFrame, cfg: Config):
    """Journalise les révisions du fournisseur ; au-delà du seuil, garde l'historique."""
    revisions: list[dict] = []
    if new.empty or old.empty or "gold_Close" not in new or "gold_Close" not in old:
        return new, revisions
    common = new.index.intersection(old.index)
    if len(common) == 0:
        return new, revisions
    a = new.loc[common, "gold_Close"].astype(float)
    b = old.loc[common, "gold_Close"].astype(float)
    ok = a.notna() & b.notna() & (b > 0)
    if not ok.any():
        return new, revisions
    diff = ((a[ok] - b[ok]).abs() / b[ok]).sort_values(ascending=False)
    big = diff[diff > cfg.max_revision_pct / 100.0]
    for dt, v in big.items():
        revisions.append({"date": f"{dt:%Y-%m-%d}", "diff_pct": float(v) * 100,
                          "old": float(b.loc[dt]), "new": float(a.loc[dt])})
    if len(big):
        for col in new.columns:
            if col in old.columns:
                new.loc[big.index, col] = old.loc[big.index, col]
    return new, revisions


def load_raw(cfg: Config) -> tuple[pd.DataFrame, list[dict]]:
    """Historique consolidé. --rebuild repart de zéro (les erreurs figées sont purgées)."""
    cache = YahooCache(cfg)
    revisions: list[dict] = []
    path = cfg.raw_file
    if path.exists() and not cfg.rebuild:
        old = normalise_daily_index(pd.read_csv(path, index_col=0, parse_dates=True))
        start = (old.index.max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        try:
            new = fetch_all(cfg, start, cache)
        except FileNotFoundError as exc:
            print(f"[DATA] {exc}")
            new = pd.DataFrame()
        if new.empty or "gold_Close" not in new or new["gold_Close"].notna().sum() == 0:
            print("[DATA] mise à jour impossible, historique conservé.")
            df = old.copy()
        else:
            new, revisions = merge_guard(new, old, cfg)
            df = new.combine_first(old).sort_index()
    else:
        if cfg.rebuild:
            print("[DATA] --rebuild : reconstruction complète de l'historique.")
        df = fetch_all(cfg, cfg.start, cache)

    if df.empty or "gold_Close" not in df.columns or df["gold_Close"].notna().sum() == 0:
        raise RuntimeError("Historique de l'or introuvable : vérifie la source et le réseau.")

    now_ny = pd.Timestamp.now(tz="America/New_York")
    df = normalise_daily_index(df)
    df = df.loc[df.index <= last_completed_market_day(now_ny, cfg)]
    df = df.dropna(subset=["gold_Close"])
    df = df[df["gold_Close"] > 0]
    if df.empty:
        raise RuntimeError("Aucune clôture d'or valide.")
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")
    print(f"[DATA] {path} | {len(df)} séances")
    return df, revisions


# ---------------------------------------------------------------- qualité ---
def scan_quality(raw: pd.DataFrame, cfg: Config) -> dict:
    """Détection seule : aucun prix n'est réécrit, aucune ligne n'est supprimée."""
    alerts, bad_bars = [], []
    if {"gold_High", "gold_Low"}.issubset(raw.columns):
        bad = (raw["gold_High"] < raw["gold_Low"]) | \
              (raw["gold_Close"] > raw["gold_High"]) | \
              (raw["gold_Close"] < raw["gold_Low"])
        for dt in raw.index[bad.fillna(False)]:
            bad_bars.append(dt)
            alerts.append({"date": f"{dt:%Y-%m-%d}",
                           "message": "barre OHLC incohérente -> range_hl/close_pos masqués"})
    recent = raw.tail(250)
    r = simple_ret(recent["gold_Close"])
    for dt, v in r[r.abs() > cfg.jump_alert_pct / 100].items():
        alerts.append({"date": f"{dt:%Y-%m-%d}",
                       "message": f"or {v * 100:+.1f} % en une séance — fait de marché à vérifier, conservé"})
    gaps = recent.index.to_series().diff().dt.days
    for dt in gaps[gaps > 6].index:
        alerts.append({"date": f"{dt:%Y-%m-%d}", "message": "trou de plus de 6 jours calendaires"})
    flat = (recent["gold_Close"].diff() == 0).astype(int)
    streak = int(flat.groupby((flat != flat.shift()).cumsum()).cumsum().max() or 0)
    if streak >= 4:
        alerts.append({"date": f"{recent.index[-1]:%Y-%m-%d}",
                       "message": f"{streak} clôtures identiques : cotation possiblement figée"})
    return {"alerts": alerts, "bad_bars": bad_bars, "flat_streak": streak}


def sanitize_external(model: pd.DataFrame, cfg: Config):
    """
    Variation SIMPLE (pas log) + confirmation par témoin.
    Un choc confirmé (ex. 30/01/2026 : argent -31 %, or -11 %) est CONSERVÉ.
    """
    out = model.copy()
    masked: dict[str, list] = {}
    events: list[dict] = []
    gld = simple_ret(out["gld_Close"]) if "gld_Close" in out.columns else None
    gold = simple_ret(out["gold_Close"]) if "gold_Close" in out.columns else None

    for col, bound in cfg.simple_move_bounds.items():
        if col not in out.columns:
            continue
        r = simple_ret(out[col])
        for dt in r.index[r.abs() > bound]:
            v = float(r.loc[dt])
            witnesses = [w for w in (gld, gold) if w is not None and pd.notna(w.get(dt, np.nan))]
            confirmed = any(abs(float(w.loc[dt])) > cfg.witness_confirm_pct / 100 for w in witnesses)
            if confirmed:
                events.append({"date": f"{dt:%Y-%m-%d}", "column": col,
                               "ret_pct": v * 100, "action": "choc confirmé par témoin : conservé"})
                continue
            out.loc[dt, col] = np.nan
            masked.setdefault(col, []).append(dt)
            events.append({"date": f"{dt:%Y-%m-%d}", "column": col, "ret_pct": v * 100,
                           "action": "non confirmé : valeur marquée inconnue"})

    if "vix_Close" in out.columns:
        rp = out["vix_Close"].astype(float).pct_change()
        for dt in rp.index[rp.abs() > cfg.vix_rel_bound]:
            out.loc[dt, "vix_Close"] = np.nan
            masked.setdefault("vix_Close", []).append(dt)
            events.append({"date": f"{dt:%Y-%m-%d}", "column": "vix_Close",
                           "ret_pct": float(rp.loc[dt]) * 100, "action": "hors borne : inconnu"})

    for dt in model.index:
        pass  # (les barres OHLC incohérentes sont masquées dans features.build)
    return out, masked, events


def audit_sources(raw: pd.DataFrame, cfg: Config) -> dict:
    """Une seule boucle pour toutes les sources secondaires, GLD inclus."""
    from . import ui
    ui.title("AUDIT MULTI-SOURCES")
    primary = safe_log_ret(raw["gold_Close"])
    start = raw.index.max() - pd.Timedelta(days=cfg.audit_window_days)
    pos = pd.Series(np.arange(len(raw.index)), index=raw.index)
    cache = YahooCache(cfg)
    rows, found = [], False

    candidates = list(cfg.audit_sources) + (["gld_witness"] if "gld_Close" in raw.columns else [])
    for src in candidates:
        if src == cfg.source:
            continue
        try:
            if src == "gld_witness":
                close = raw["gld_Close"]
            elif src == "yahoo":
                close = cache.download("GC=F", start.strftime("%Y-%m-%d")).get("Close")
            elif src == "netdania_csv" and cfg.netdania_csv.exists():
                close = read_local_csv(cfg.netdania_csv, "NetDania", cfg, True,
                                       cfg.netdania_tz, False)["Close"]
            elif src == "google_csv" and cfg.google_csv.exists():
                close = read_local_csv(cfg.google_csv, "Google", cfg, verbose=False)["Close"]
            elif src == "alphavantage" and cfg.alphavantage_csv.exists():
                close = read_local_csv(cfg.alphavantage_csv, "AlphaVantage", cfg, verbose=False)["Close"]
            else:
                continue
        except Exception as exc:
            print(f"[AUDIT] {src} indisponible : {exc}")
            continue
        if close is None or len(close) == 0:
            continue
        alt = safe_log_ret(close.reindex(raw.index))
        both = pd.concat([primary.rename("p"), alt.rename("a")], axis=1).dropna()
        if both.empty:
            continue
        step = pos.reindex(both.index).diff().eq(1)
        both = both.loc[step | (both.index == both.index[0])]
        both = both.loc[both.index >= start]
        if len(both) < 60:
            continue
        found = True
        corr = {lag: float(both["p"].shift(lag).corr(both["a"])) for lag in range(-2, 3)}
        corr = {k: (v if np.isfinite(v) else -1.0) for k, v in corr.items()}
        best = max(corr, key=corr.get)
        last = abs(float(both["p"].iloc[-1] - both["a"].iloc[-1])) * 100
        rows.append({"Source": src, "n": str(len(both)), "corr lag0": fr_(corr[0]),
                     "meilleur lag": f"{best:+d}", "écart dernier": f"{last:.2f} %"})
        if best != 0 and corr[best] > corr[0] + 0.05:
            print(f"[AUDIT] {src} : décalage de dates probable (meilleur lag {best:+d}).")
        if last > cfg.divergence_warn_pct:
            print(f"[AUDIT] {src} : divergence récente {last:.2f} %.")

    if rows:
        print(ui.md(pd.DataFrame(rows), align=("left", "right", "center", "center", "right")))
    else:
        print("Aucune source secondaire exploitable.")
    return {"available": rows, "found": found}


def fr_(v):  # petit utilitaire local, évite une dépendance circulaire au formatage
    return f"{v:.3f}" if v is not None and math.isfinite(v) else "—"