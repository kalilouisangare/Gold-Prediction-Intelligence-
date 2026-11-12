# Gold Daily V7 — moteur quantitatif honnête

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/code-MIT-green)
![Status](https://img.shields.io/badge/statut-recherche-orange)

Moteur de prévision **par intervalles** sur l'or quotidien (GC=F, XAU/USD), conçu autour d'une
règle simple : **ne jamais annoncer un avantage qui n'a pas été mesuré**.

Le système ne prédit pas la direction. Il produit des fourchettes de probabilité dont la
couverture est **testée statistiquement**, et une couche de risque bornée par construction.

> ⚠️ **Avertissement.** Projet de recherche à but éducatif. Ne constitue pas un conseil
> d'investissement. Le script ne passe aucun ordre et ne se connecte à aucun courtage.
> Une fourchette à 80 % signifie que 2 clôtures sur 10 tomberont **hors** de l'intervalle.

---

## Table des matières

- [Philosophie](#philosophie)
- [Ce que V7 corrige par rapport à V6](#ce-que-v7-corrige-par-rapport-à-v6)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Routine quotidienne](#routine-quotidienne)
- [Tests](#tests)
- [Architecture](#architecture)
- [Méthodologie](#méthodologie)
- [Qualité des données : la doctrine](#qualité-des-données--la-doctrine)
- [Couche risque](#couche-risque)
- [Lire les résultats](#lire-les-résultats)
- [Glossaire](#glossaire)
- [Fichiers produits](#fichiers-produits)
- [Résultats observés](#résultats-observés)
- [Limites connues](#limites-connues)
- [Roadmap](#roadmap)
- [Contribuer](#contribuer)
- [Données, secrets et licence](#données-secrets-et-licence)

---

## Philosophie

Trois engagements, appliqués dans le code :

1. **Aucun avantage inventé.** Un modèle n'est retenu que s'il bat **deux** baselines
   (`naive` = prix inchangé, et `drift` = momentum 60 séances), avec une p-value corrigée
   pour le nombre de candidats, **puis** confirmé sur un bloc chronologique indépendant.
   Sinon : retour à `naive`, centre = dernière clôture, verdict « aucun avantage mesuré ».
2. **Aucune donnée inventée.** Un mouvement extrême confirmé par un témoin est un fait de
   marché : il est conservé. Seule une valeur *non confirmée* est marquée inconnue (`NaN`),
   jamais remplacée par une estimation.
3. **Aucune sécurité non mesurée.** La couverture des fourchettes est vérifiée en continu
   sur les prévisions réellement échues (tests de Kupiec et Christoffersen), avec
   coupe-circuit automatique en cas de rejet.

---

## Ce que V7 corrige par rapport à V6

| # | Correctif | Module |
|---|---|---|
| 1 | Bornes de mouvement exprimées en **variation simple** (et non log) + **confirmation par témoin** : un krach réel n'est plus effacé | `data.py` |
| 2 | **Quarantaine réelle** des barres OHLC incohérentes (masquage des features, pas d'une copie jetée) | `data.py`, `features.py` |
| 3 | Garde-fou de fusion : révisions **journalisées**, `--rebuild` pour purger un historique figé | `data.py` |
| 4 | `EXTERNAL_LAG = 1` : fin de la fuite liée aux horaires de clôture (or 13h30 ET vs VIX 16h15 ET) | `config.py` |
| 5 | **Conformal adaptatif** persistant + couverture **par régime de volatilité** | `calibration.py` |
| 6 | Tests de VaR **Kupiec + Christoffersen** sur prévisions **non chevauchantes** | `calibration.py`, `tracking.py` |
| 7 | Couche décision **atteignable** : P(gain > coûts), stop sur borne 90 %, taille bornée par le risque, **coupe-circuit** | `risk.py` |
| 8 | Journal par **date cible** + table « même échéance, information actualisée » | `tracking.py` |
| 9 | `n_candidates` déduit du registre (plus de constante en dur), MLP/xgb optionnels | `models.py` |
| 10 | Rapport JSON écrit **avant** l'affichage ; l'affichage ne peut plus faire échouer un run | `engine.py`, `ui.py` |

---

## Installation

Prérequis : **Python ≥ 3.10** (développé et testé sur 3.14).

```bash
git clone https://github.com/TON_USER/TON_REPO.git
cd TON_REPO

# avec uv (recommandé)
uv venv
uv pip install -r requirements.txt

# ou avec pip
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

`tabulate` est **optionnel** : sans lui, les tables basculent automatiquement sur un rendu
ASCII intégré. `scipy` n'est **pas** requis (le test du χ² à 1 ddl utilise `math.erfc`).
`xgboost` est optionnel : sans lui, le registre se rabat sur ridge seul.

Clé Alpha Vantage (facultative, uniquement pour la source `alphavantage`) :

```bash
cp .env.example .env      # puis renseigner ALPHAVANTAGE_API_KEY
export ALPHAVANTAGE_API_KEY="ta_clé"
```

---

## Utilisation

Le module se lance **depuis la racine du dépôt**, jamais depuis l'intérieur du dossier
`gold_daily_v7/` (les imports sont relatifs).

```bash
python -m gold_daily_v7 --help                       # vérifie l'installation, sans réseau
python -m gold_daily_v7                              # run standard (source Yahoo GC=F)
python -m gold_daily_v7 3975.50                      # + fourchettes recentrées sur un spot
python -m gold_daily_v7 --source netdania_csv        # source alternative
python -m gold_daily_v7 --rebuild                    # reconstruit tout l'historique
python -m gold_daily_v7 --equity 25000 --risk 0.4    # paramètres de risque
python -m gold_daily_v7 --mlp                        # réactive les candidats réseaux de neurones
```

| Option | Défaut | Effet |
|---|---|---|
| `spot_or_source` | — | Prix spot (`3975.50` / `3975,50`) **ou** nom de source |
| `--source` | `yahoo` | `yahoo`, `netdania_csv`, `google_csv`, `alphavantage` |
| `--no-cache` | off | Ignore le cache disque |
| `--refresh` | off | Force le retéléchargement |
| `--rebuild` | off | Repart de zéro (purge les barres figées par le garde-fou) |
| `--mlp` | off | Ajoute `mlp` et `mlp_deep` aux candidats |
| `--no-xgb` | off | Retire xgboost (et donc `blend`) |
| `--equity` | `10000` | Capital de référence pour le sizing |
| `--risk` | `0.5` | Pourcentage du capital risqué par position |
| `--prob-min` | `0.60` | P(gain > coûts) minimale pour émettre un signal |
| `--data-dir` | `gold_data/v7` | Racine des sorties |

Sources CSV attendues (colonnes `Date` + `Close` minimum, OHLCV reconnu souplement,
intraday reconstruit à la coupure 17h New York) :

```
data/raw/netdania/XAUUSD.csv
data/raw/google_finance/XAUUSD.csv
data/raw/alphavantage/XAUUSD.csv
```

---

## Routine quotidienne

```bash
cd chemin/vers/le/depot
source .venv/bin/activate
python -m gold_daily_v7
```

**Sans `--rebuild`.** Le run quotidien ajoute la séance, fait mûrir les prévisions, nourrit
le conformal adaptatif et relance les tests de couverture.

| Quand | Commande |
|---|---|
| Chaque jour de bourse | `python -m gold_daily_v7` |
| Cache suspecté périmé | `python -m gold_daily_v7 --refresh` |
| Après un incident données uniquement | `python -m gold_daily_v7 --rebuild` |
| Jamais | supprimer `conformal_state_v7.json` |

Chaque run refait le walk-forward complet (8 folds × 3 horizons) : compte quelques minutes.
Il n'y a pas de mode incrémental, et c'est voulu — la fenêtre hors échantillon glisse d'une
séance chaque jour, donc le quantile doit être réestimé.

**Créneau horaire.** Le code n'admet que les séances raisonnablement terminées
(`heure de New York − 17h30`). Deux usages cohérents :

- matin Paris (07h–09h) → ancre = clôture US de la veille, briefing sur les séances à venir ;
- soir Paris (après 23h30) → ancre = clôture US du jour, information la plus fraîche.

Choisis-en un et tiens-y toi. Deux runs avec la même ancre sont idempotents : la
déduplication du journal garde la première prévision.

Avec log horodaté :

```bash
mkdir -p logs
python -m gold_daily_v7 2>&1 | tee "logs/v7_$(date +%F).log"
```

Automatisation possible par cron (`0 7 * * 2-6`, mardi→samedi pour couvrir les séances US
du lundi→vendredi). Sous WSL, cron ne tourne que si la distribution est démarrée ; sinon
utiliser le Planificateur de tâches Windows. Dans tous les cas, **lis la sortie** : un moteur
dont le principe est « aucun avantage sans preuve » ne se pilote pas en aveugle.

**Sauvegarde.** Deux fichiers sont irremplaçables : `forecasts_log_v7.csv` (base de tous les
tests) et `conformal_state_v7.json` (mémoire de la calibration). Sauvegarde-les chaque
semaine, hors du dépôt. Le reste se reconstruit.

---

## Tests

Test hors-ligne complet, aucun appel réseau (~30 s) :

```bash
python selftest.py
```

Il déroule deux runs bout en bout sur données synthétiques et vérifie :

- la conservation d'un krach confirmé par témoin ;
- le masquage d'un spike isolé non confirmé ;
- la quarantaine des barres OHLC incohérentes ;
- les tests de Kupiec et Christoffersen sur des cas connus ;
- le **sens** de mise à jour du conformal adaptatif (ratés → élargit, zéro raté → rétrécit) ;
- l'écriture du rapport, du journal et de l'état persistant.

Variante avec xgboost et le composite `blend` :

```bash
sed 's/use_xgb=False/use_xgb=True/' selftest.py > selftest_xgb.py
python selftest_xgb.py
```

Les tests de couverture sur données réelles apparaissent dès **20 prévisions non
chevauchantes échues** ; le coupe-circuit, lui, exige **n ≥ 100**.

---

## Architecture

```
.
├── selftest.py           # test hors-ligne bout en bout
├── requirements.txt
├── README.md
└── gold_daily_v7/
    ├── __init__.py
    ├── __main__.py       # CLI, argparse, warnings
    ├── config.py         # un seul objet Config : tous les paramètres et chemins
    ├── ui.py             # formatage FR + tables (tabulate optionnel, repli ASCII)
    ├── data.py           # sources, cache, store, garde-fou, qualité, audit
    ├── features.py       # 30+ variables, masquage qualité, couverture
    ├── models.py         # registre, walk-forward purgé, bootstrap, sélection
    ├── calibration.py    # conformal, adaptatif, régimes, Kupiec/Christoffersen
    ├── risk.py           # décision, sizing, coupe-circuit
    ├── tracking.py       # journal par date cible, échéances, réconciliation
    └── engine.py         # orchestration + rapport JSON
```

Aucune dépendance circulaire : `config` ← `data`/`features`/`models` ← `calibration`/`risk`
← `tracking` ← `engine`. `ui` est transversal et ne lève jamais : un échec d'affichage ne
peut pas coûter un run.

---

## Méthodologie

### Validation chronologique

- **Walk-forward** : 8 folds × 250 séances, entraînement ≥ 500 lignes.
- **Purge** : une observation d'entraînement n'est conservée que si sa cible se termine
  **avant** le début du fold de test (`target_end_pos < start_pos`). Pas de fuite par
  chevauchement de cibles.
- **Découpage OOS** : 50 % sélection / 25 % confirmation / 25 % calibration.
- **Double baseline** : `naive` et `drift` ne sont jamais candidates.
- **Correction de multiplicité** : seuil `α / n_candidates` en sélection, `α / 2` en
  confirmation (deux baselines). Avec ridge + xgb + blend, le seuil de sélection vaut
  0,10 / 3 = **0,033**.
- **Bootstrap par blocs** (20 séances, 2 000 tirages) pour les p-values : les cibles
  multi-séances sont autocorrélées, un bootstrap i.i.d. serait frauduleux.

### Intervalles

1. Résidus standardisés `z = (y − μ) / (vol_ewma · √h)`.
2. **Conformal split** : `q = quantile(|z|, 1 − α)` sur le bloc de calibration.
3. **Conformal adaptatif** (Gibbs & Candès) : le niveau `α_t` est mis à jour en ligne à
   partir des résidus réellement observés, `α_{t+1} = α_t + γ(α − err_t)`. L'état est
   persisté dans `conformal_state_v7.json` et ne doit pas être supprimé entre deux runs.
   Tant que moins de **60 observations live** sont disponibles, `q` reste égal au `q` du
   conformal split : aucune adaptation sans preuve.
4. **Couverture et gain de Winkler** mesurés sur une moitié tenue à l'écart, jamais
   en échantillon.
5. **Audit par régime** : la couverture est aussi rapportée par tercile de volatilité —
   une couverture globale correcte peut cacher 85 % en calme et 65 % en stress.

### Tests de couverture (sur prévisions échues)

| Test | Hypothèse nulle | Ce qu'il détecte |
|---|---|---|
| **Kupiec** (POF) | Le taux de sortie vaut `1 − niveau` | Fourchette trop étroite ou trop large |
| **Christoffersen** | Les sorties sont indépendantes | Ratés **groupés** (le vrai risque en trading) |

Les tests ne portent que sur des prévisions **non chevauchantes** (une tous les `h` jours
d'ancrage) : sinon l'autocorrélation mécanique des cibles biaise le verdict.

---

## Qualité des données : la doctrine

**Un mouvement extrême n'est pas une erreur.**

Le 30 janvier 2026, les futures argent ont chuté de **31,4 %** (règlement 78,53 $, pire
séance depuis mars 1980) et l'or de **11,4 %** (règlement 4 745,10 $), après la nomination
de Kevin Warsh à la Fed
([CNBC](https://www.cnbc.com/2026/01/30/silver-gold-fall-price-usd-dollar-fed-warsh-chair-trump-metals.html)).

La V6 traitait cet argent comme une donnée corrompue : sa borne « physique » de 30 % était
comparée à un rendement **logarithmique** (`−0,376` en log = `−31,3 %` en réel), et aucune
confirmation par témoin n'était demandée. Résultat : jusqu'à 9 séances de features
détruites autour de l'événement le plus informatif de l'échantillon.

V7 applique trois règles :

1. Les bornes sont en **variation simple**, jamais en log.
2. Un choc n'est marqué inconnu que si **aucun témoin** (GLD, or) ne bouge de plus de 5 %.
3. La détection **alerte**, elle ne bloque pas et ne réécrit rien.

### Les quatre niveaux de la table qualité

| Niveau | Signification | Action sur les données |
|---|---|---|
| **ALERTE** | Anomalie repérée, à vérifier par un humain | Aucune, sauf pour les barres OHLC incohérentes dont `range_hl` / `close_pos` sont masqués |
| **CONSERVÉ** | Mouvement extrême **confirmé** par un témoin | Aucune — c'est un fait de marché |
| **MASQUÉ** | Mouvement extrême **non confirmé** | Valeur mise à `NaN`, jamais remplacée par une estimation |
| **RÉVISION** | Le fournisseur a modifié une clôture passée de plus de 2 % | Ancienne valeur conservée, écart journalisé |

Le fichier brut est un historique **consolidé**, pas une archive immuable : les révisions
fournisseur ≤ 2 % sont acceptées, les autres sont journalisées et l'historique est conservé.
`--rebuild` permet de purger une valeur figée à tort.

Témoin gratuit : **GLD** (ETF physique), indépendant du contrat futures GC=F, plus un
contrôle de cohérence macro (DXY, pétrole, taux, VIX).

---

## Couche risque

Un signal directionnel n'est émis que si **les trois** conditions sont réunies :

1. Le modèle a passé sélection **et** confirmation (`validated = True`).
2. `P(rendement > coûts) ≥ prob_min`, calculée sur la **loi empirique** des résidus
   standardisés (pas d'hypothèse gaussienne).
3. Le **coupe-circuit** est inactif.

Le sizing n'est jamais fonction de la conviction, seulement du risque :

```
distance_stop = |p0 − borne_90| / p0
notional_max  = equity × risk_pct / distance_stop
```

Exemple issu du run du 12/11/2026 (h = 1) : capital 10 000 $, risque 0,5 % = 50 $,
stop sur la borne 90 % à 3 901,3 $ contre un prix de 4 000,5 $, soit 2,48 % de distance
→ notional max **2 016 $**, soit 0,504 once, levier 0,20×. Une fourchette ratée ne peut
pas ruiner le compte.

**Coupe-circuit** : si Kupiec rejette la couverture à 5 % sur ≥ 100 prévisions échues et
que la couverture observée est inférieure au niveau annoncé, tout passe en `NEUTRAL`
jusqu'à recalibration.

---

## Lire les résultats

### Une fourchette n'est pas un objectif de prix

C'est un triplet **(date d'ancrage, date cible, niveau de confiance)**. Exemple illustratif :

| Émis le | Ancre | h | Cible | Fourchette 80 % |
|---|---|---|---|---|
| 10/11 | 4 060 | 2 | **12/11** | 3 976 – 4 145 |
| 11/11 | 3 995 | 1 | **12/11** | 3 949 – 4 041 |
| 11/11 | 3 995 | 2 | 13/11 | 3 930 – 4 061 |

« Séance 2 » le 10/11 vise le 12/11 ; « séance 2 » le 11/11 vise le 13/11. **Ce ne sont pas
les mêmes questions.** Pour comparer, fixez la date cible — c'est exactement ce que fait
la table « même échéance, information actualisée ».

### Pourquoi la largeur change

Le centre suit la dernière clôture (modèle `naive`). La largeur suit `vol_ewma`, dont la
demi-vie est de **11,2 séances** : un seul rendement de −11,4 % fait passer la vol de 1,00 %
à 2,96 %, et l'effet met environ deux mois à s'estomper. Des fourchettes qui bougent sont
le signe d'un système qui réagit, pas d'un système incohérent.

### Pourquoi « retard 1 j » est normal

Le code n'admet que les séances raisonnablement terminées (`heure de New York − 17h30`),
puis compte les jours calendaires jusqu'à la dernière clôture retenue. Un run lancé le
12/11 avant 17h30 à New York s'ancre sur le 11/11 : la séance du 12/11 n'a pas encore de
règlement. `retard = 1` est le fonctionnement attendu. Le garde-fou ne refuse de travailler
qu'au-delà de **7 jours**, c'est-à-dire si la source est réellement en panne.

### Exemple de sortie (run du 12/11/2026)

```
=== LECTURE SIMPLE | YAHOO ===
Dernière clôture : 11/11/2026  →  4 000,5 $
╭───────────┬────────────┬───────────┬───────────────────┬──────────╮
│  Horizon  │  Échéance  │    Centre │  Fourchette 80 %  │  ± 80 %  │
├───────────┼────────────┼───────────┼───────────────────┼──────────┤
│    1 j    │   12/11    │ 4 000,5 $ │ 3 927,5 – 4 074,8 │ ± 1,8 %  │
│    2 j    │   13/11    │ 4 000,5 $ │ 3 897,9 – 4 105,8 │ ± 2,6 %  │
│    5 j    │   18/11    │ 4 000,5 $ │ 3 836,7 – 4 171,3 │ ± 4,2 %  │
╰───────────┴────────────┴───────────┴───────────────────┴──────────╯
```

---

## Glossaire

### Les modèles

| Nom | Ce qu'il prédit | Rôle |
|---|---|---|
| **naive** | « le prix ne bouge pas » → μ = 0, centre = dernière clôture | **Baseline**, jamais candidat. Hypothèse du marché aléatoire |
| **drift** | Momentum moyen des 60 dernières séances × h | **Baseline** « tendance », jamais candidat |
| **ridge** | Régression linéaire sur les variables, pénalité L2 (`alpha = 2000`, très forte) | Candidat |
| **xgb** | Arbres de décision boostés (xgboost) | Candidat |
| **blend** | Moyenne des prédictions de ridge et de xgb | Candidat |
| **mlp**, **mlp_deep** | Réseaux de neurones tabulaires | Désactivés par défaut (`--mlp`) |

`alpha = 2000` écrase les coefficients de ridge vers zéro : avec ~2 000 points et 32
variables bruitées, une régularisation faible surajusterait. C'est pourquoi ridge et naive
ont des erreurs quasi identiques.

### La p-value

**p = probabilité d'observer un gain au moins aussi grand si le modèle n'a en réalité
aucun avantage.** Une p-value élevée means « résultat banal sous l'hypothèse d'absence
d'avantage », donc aucune preuve.

Trois pièges :

- p n'est **pas** la probabilité que le modèle soit bon ;
- un gain positif avec p élevée n'est pas un avantage mesuré ;
- l'étoile `*` signale p < seuil corrigé. Plus on ajoute de candidats, plus le seuil est
  dur : c'est la protection contre « j'ai essayé 50 choses, une a marché par chance ».

Autres colonnes des tables de sélection : **Erreur %** = erreur absolue moyenne en log
×100 ; **Erreur $** = la même en dollars, moyennée sur la fenêtre hors échantillon (donc à
des niveaux d'or inférieurs au prix actuel) ; **Direction %** = part des séances où le
signe prédit est le bon (50 % = pile ou face).

### Régime de volatilité

La volatilité mesure l'amplitude typique des variations quotidiennes, estimée ici par
`vol_ewma` (moyenne mobile exponentielle des carrés de rendements, demi-vie 11,2 séances).
Les séances testées sont réparties en **trois tiers égaux** selon cette volatilité :
« vol basse », « vol moyenne », « vol haute ». La couverture est mesurée dans chacun.

Une couverture globale de 80 % peut cacher 71 % en calme et 85 % en stress : on serait mal
protégé exactement au changement de régime.

### « Couverture » : deux sens à ne pas confondre

**Couverture des variables** — table `COUVERTURE DES VARIABLES` : fraction des séances pour
lesquelles la variable est calculable. Elle se vérifie arithmétiquement :

| Variable | Attendu | Observé |
|---|---|---|
| `dist_ma200` | 200 séances d'amorçage → 200/4241 = 4,7 % manquants | 95,3 % |
| `ret_60` | 60/4241 = 1,4 % | 98,6 % |
| `vol_20` | 20/4241 = 0,5 % | 99,5 % |
| `mom_drift` | 30 séances minimum | 99,3 % |

Une variable sous 90 % de couverture est écartée (`min_feature_coverage`).

**Couverture des fourchettes** — tables `HORIZON n` : fraction des séances où le prix réel
est tombé dans l'intervalle annoncé. C'est la mesure de fiabilité du système.

---

## Fichiers produits

| Chemin | Contenu | Versionné ? |
|---|---|---|
| `gold_data/v7/raw/raw_prices_*.csv` | Historique consolidé (source de vérité locale) | ❌ |
| `gold_data/v7/cache/*.csv` | Cache Yahoo (TTL 6 h) | ❌ |
| `gold_data/v7/forecasts_log_v7.csv` | Journal dédupliqué par (ancre, horizon, version, source) | ❌ |
| `gold_data/v7/report_v7.json` | Rapport complet pour dashboard | ❌ |
| `gold_data/v7/oos_predictions_v7.csv` | Prévisions hors échantillon (graphiques sans recalcul) | ❌ |
| `gold_data/v7/conformal_state_v7.json` | État du conformal adaptatif — **à conserver** | ❌ |
| `logs/` | Logs horodatés des runs | ❌ |

Schéma du rapport :

```json
{
  "model_version": "7.0",
  "generated_at": "...",
  "data": { "start": "...", "end": "...", "n_sessions": 4241 },
  "quality": { "alerts": [], "masked": [], "revisions": [] },
  "audit": { "available": [] },
  "var_tests": [
    { "horizon": 1, "level": 80, "n": 240, "coverage": 79.6,
      "kupiec_p": 0.87, "christoffersen_p": 0.41, "verdict": "conforme" }
  ],
  "conformal_state": { "1_80": { "alpha_t": 0.20, "n_live": 0, "base_q": 1.257 } },
  "horizons": {
    "1": { "model": "naive", "validated": false, "bands": {}, "levels": {}, "signal": {} }
  }
}
```

---

## Résultats observés

Run V7 du 12/11/2026, source Yahoo GC=F, 4 241 séances (2010-01-04 → 2026-11-11),
8 folds × 250 séances, 32 variables, `EXTERNAL_LAG = 1`.

**Aucun modèle ne bat les deux baselines.** Sélection à h = 1 (seuil corrigé α/3 = 0,033) :

| Modèle | Erreur (log %) | Gain vs naive | p vs naive | Gain vs drift | p vs drift | Direction |
|---|---|---|---|---|---|---|
| naive (réf.) | 0,73 | — | — | +0,68 % | 0,054 | — |
| ridge | 0,73 | −0,45 % | 0,863 | +0,24 % | 0,342 | 51,2 % |
| xgb | 0,73 | −1,02 % | 0,948 | −0,33 % | 0,657 | 51,1 % |
| blend | 0,73 | −0,60 % | 0,898 | +0,09 % | 0,438 | 50,2 % |
| drift (réf.) | 0,73 | −0,69 % | 0,947 | — | — | 51,6 % |

Modèle retenu : `naive` aux trois horizons.

**Couverture tenue à l'écart :**

| Horizon | 80 % observé | 90 % observé | Demi-largeur 80 % | Gain de Winkler 80 % |
|---|---|---|---|---|
| 1 | 78,3 % | 91,7 % | ± 1,35 % | +5,9 % |
| 2 | 77,9 % | 89,9 % | ± 1,90 % | +6,4 % |
| 5 | 79,7 % | 91,6 % | ± 3,06 % | +2,2 % |

Le ratio empirique `q90/q80` vaut 1,36 à 1,41 selon l'horizon, contre 1,28 pour une
gaussienne : les queues épaisses sont bien captées.

**Couverture 80 % par régime de volatilité :**

| Horizon | vol basse | vol moyenne | vol haute |
|---|---|---|---|
| 1 | 76,8 % | 74,6 % | 80,6 % |
| 2 | 73,4 % | 74,6 % | 81,1 % |
| 5 | 70,9 % | 75,0 % | 85,2 % |

L'ordre basse < moyenne < haute se répète aux trois horizons. Aucune cellule ne rejette
individuellement une fois l'effectif corrigé du chevauchement des cibles (IC 95 % de ±5 à
±12 points). Hypothèse retenue : `vol_ewma` ne prend pas en compte la révision à la moyenne
de la volatilité. Correctif à l'étude : conformal conditionnel par régime, puis prévision
HAR/GARCH.

**Qualité des données :** le krach du 30/01/2026 (or −10,8 % en variation simple, argent
−31,4 %) est détecté, confirmé par le témoin GLD et **conservé**. 25 barres OHLC
incohérentes (2010-2011) sont quarantainées ; hors fenêtre hors échantillon, sans impact
sur la calibration. Audit GLD : 746 séances communes, corrélation lag0 = 0,916.

**Conclusion assumée : la direction n'est pas prédictible ici. La couverture, si.**
Les tests de Kupiec et Christoffersen n'apparaissent qu'après suffisamment de prévisions
réellement échues. Ces chiffres décrivent un run passé et ne préjugent d'aucun résultat
futur.

---

## Limites connues

- **GC=F est un contrat continu non ajusté** : les changements d'échéance créent des sauts
  de prix. Ils sont détectés et signalés, jamais réécrits. Une série back-adjusted ou un
  spot XAU/USD propre reste préférable.
- **Biais haussier latent de la couche décision** : la loi empirique des résidus est centrée
  sur le drift réalisé de l'échantillon, pas sur zéro. Sur le run du 12/11/2026,
  P(hausse > coûts) = 55,6 % contre P(baisse > coûts) = 40,1 % à h = 5, alors que le modèle
  retenu est `naive` (μ = 0). Aucun signal n'est émis aujourd'hui, mais si un modèle était
  validé, la décision serait mécaniquement biaisée vers l'achat. Correctif prévu : centrer
  la loi empirique avant le calcul des probabilités.
- **Warm-up du conformal adaptatif** : tant que moins de 60 prévisions réellement échues
  ont été observées, `q retenu = q base`. L'adaptation ne démarre qu'ensuite — c'est voulu.
- **Puissance statistique faible** : à h = 5, 2 000 observations hors échantillon ne valent
  qu'environ 400 blocs indépendants. Ne pas surinterpréter une p-value isolée.
- **Horizons chevauchants** : corrigé dans les tests, mais la couverture effective reste
  plus incertaine à h = 5 qu'à h = 1.
- **Révisions fournisseur** : Yahoo peut modifier une clôture passée. Deux runs à dates
  égales peuvent donc différer légèrement. `--rebuild` casse la reproductibilité.
- **Dépendance réseau** : Yahoo peut limiter ou renvoyer des séries vides ; repli sur cache.
- **Aucun backtest de portefeuille** : le sizing est une règle de risque, pas une simulation
  de P&L avec frais, slippage réel et financement.

---

## Roadmap

- [ ] Centrer la loi empirique des résidus pour supprimer le biais directionnel latent
- [ ] Conformal conditionnel par régime (Mondrian), validé par comparaison avant/après
- [ ] Prévision de volatilité HAR-RV / GARCH au lieu d'un EWMA purement réactif
- [ ] Série or back-adjusted (détection de roll par volume/open interest) ou spot XAU/USD
- [ ] Backtest de portefeuille avec frais, slippage et financement
- [ ] Filtre événements macro (FOMC, CPI, NFP) en no-trade
- [ ] Dashboard Streamlit branché sur `report_v7.json` et `oos_predictions_v7.csv`
- [ ] Tests unitaires (purge du walk-forward, Kupiec, conformal adaptatif)
- [ ] CI GitHub Actions : lint + tests sur 3.10 → 3.14

---

## Contribuer

Les issues et PR sont bienvenues, en particulier sur la calibration conditionnelle et la
qualité des données. Merci de garder la doctrine : **aucun avantage annoncé sans preuve,
aucune donnée inventée**. Toute modification de la calibration doit être accompagnée d'une
comparaison de couverture sur bloc tenu à l'écart, et `python selftest.py` doit rester vert.

---

## Données, secrets et licence

- **Le code est sous licence MIT** (voir `LICENSE`).
- **Les données de marché ne sont pas incluses** et ne doivent pas être republiées :
  respectez les conditions d'utilisation de Yahoo Finance, NetDania, Google Finance et
  Alpha Vantage. Le dossier `gold_data/` est ignoré par git.
- **Aucune clé API n'est versionnée.** `ALPHAVANTAGE_API_KEY` est lue dans l'environnement ;
  `.env` est ignoré.
- Le journal de prévisions est personnel : il n'a pas vocation à être publié.