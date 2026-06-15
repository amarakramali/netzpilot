# NetzPilot - Prognose-Engine v1 (Vertical Slice)

Day-ahead-Lastprognose auf **echten** SMARD-Daten (Realisierter Stromverbrauch / Netzlast DE,
stuendlich, 01.01.-24.03.2024, 2016 Stunden, lueckenlos).

## Lauf
```
python3 forecast_pipeline.py     # benoetigt nur numpy + pandas
```
Erzeugt `forecast_results.json` (Metriken) und `forecast_arrays.npz` (Backtest-Arrays).

## Methode
- Ziel: stuendliche Netzlast, Horizont Day-ahead 24h (Stichtag 00:00 lokal).
- Baselines: Persistenz (t-24h), Saisonal-Naiv (t-168h).
- Modell: Saisonal-Naiv + Ridge-Korrektur der Wochenabweichung (P50); P10/P90 ueber
  stundenbedingte Residuenquantile.
- Validierung: leakage-sicheres Rolling-Origin-Backtest (letzte 28 Tage, taegliches Retraining).
  Alle Features sind zum Stichtag bekannt (Lags >=24h, Vortagesstatistik, Kalender/Fourier).

## Ergebnis (Test: 28 Tage / 672 Stunden)
| Verfahren | MAE [MW] | MAPE [%] | MASE | Skill vs Pers. | Skill vs S-Naiv |
|---|---|---|---|---|---|
| Persistenz | 3816 | 6.93 | 0.995 | 0% | -159% |
| Saisonal-Naiv | 1472 | 2.71 | 0.384 | 61% | 0% |
| NetzPilot (Ridge) | **1411** | **2.56** | **0.368** | **63%** | **+4.1%** |

P10-P90-Abdeckung: 81.5 % (Ziel 80 %) -> kalibrierte Unsicherheit.

## Ehrliche Einordnung
- Nationale Last ist glatter als die eines kleinen Stadtwerks (dort hoehere MAPE erwartbar).
- Bewusst transparentes lineares Modell (Build-Umgebung ohne LightGBM/Internet).
  Upgrade-Pfad: LightGBM-Quantilregression + Wetter-Features (Open-Meteo) -> weitere Verbesserung.
- Datenfenster 12 Wochen; Skalierung auf Mehrjahres-/15-min-Daten ist mechanisch.

## Datenquelle / Lizenz
SMARD.de (Bundesnetzagentur), Realisierter Stromverbrauch, Filter 410, Region DE, stuendlich.
Lizenz: CC BY 4.0 ("Bundesnetzagentur | SMARD.de").
