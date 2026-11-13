import pandas as pd

d = pd.read_csv("gold_data/v7/raw/raw_prices_xauusd_spot_investing.csv",
                index_col=0, parse_dates=True)

print("Debut      :", d.index.min().date(), d.index.min().day_name())
print("Fin        :", d.index.max().date(), d.index.max().day_name())
print("Seances    :", len(d))
print("Week-ends  :", int((d.index.day_of_week >= 5).sum()))
print("1er janvier:", int(((d.index.month == 1) & (d.index.day == 1)).sum()))
print("25 decembre:", int(((d.index.month == 12) & (d.index.day == 25)).sum()))
print("Colonnes   :", list(d.columns))

diff = d["gold_Close"].diff()
flat = (diff == 0).astype(int)
streak = int(flat.groupby((flat != flat.shift()).cumsum()).cumsum().max())
print("Clotures identiques consecutives (max) :", streak)

print()
print("NaN par colonne :")
print(d.isna().sum())

print()
print(d.head(3))
print(d.tail(3))
