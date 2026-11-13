import pandas as pd
from gold_daily_v7.config import Config
from gold_daily_v7 import data, features

raw = pd.read_csv("gold_data/v7/raw/raw_prices_xauusd_spot_investing.csv",
                  index_col=0, parse_dates=True)
cfg = Config(source="investing")
model, masked, events = data.sanitize_external(raw, cfg)
q = data.scan_quality(raw, cfg)
f, cov = features.build(model, cfg, masked, q["bad_bars"])

rows = []
for c in f.columns:
    s = f[c].dropna()
    rows.append({"variable": c, "n": len(s),
                 "part_zeros": (s == 0).mean(), "std": s.std()})
t = pd.DataFrame(rows).sort_values("part_zeros", ascending=False)
print(t.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
