import pandas as pd

d = pd.read_csv("gold_data/v7/raw/raw_prices_xauusd_spot_investing.csv",
                index_col=0, parse_dates=True)

gap = d["gold_Open"] / d["gold_Close"].shift(1) - 1
gap = gap.dropna()

print("gap_open -- moyenne |gap| :", f"{gap.abs().mean():.8f}")
print("gap_open -- ecart-type    :", f"{gap.std():.8f}")
print("part |gap| < 0.01 %       :", f"{(gap.abs() < 1e-4).mean():.1%}")
print("part gap exactement nul   :", f"{(gap == 0).mean():.1%}")
print()
print("Moyenne |gap| par annee :")
print(gap.abs().groupby(gap.index.year).mean().round(8))
