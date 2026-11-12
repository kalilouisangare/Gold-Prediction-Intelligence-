cd ~/HOME/Data_Science/Gold_trading
source .venv/bin/activate
python -m gold_daily_v7


""" ----------------------------------------- """
python -m gold_v7                      # run standard
python -m gold_v7 3975.50              # + fourchettes recentrées sur le spot
python -m gold_v7 --source netdania_csv
python -m gold_v7 --rebuild            # purge les erreurs figées par le garde-fou V6
python -m gold_v7 --equity 25000 --risk 0.4 --prob-min 0.65
python -m gold_v7 --mlp                # si tu veux re-tester les réseaux