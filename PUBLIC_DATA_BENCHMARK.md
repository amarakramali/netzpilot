# NetzPilot — öffentliche DSO-Daten-Baseline

*Stand: 2. Juli 2026. Abgeleitete Kennzahlen; keine Rohdaten werden in diesem Repository
weiterverteilt.*

## Ergebnis

Der konservative NetzPilot-Champion (`ShrunkCorrector`, 84 rollierende Testtage,
10.000 paired Block-Bootstrap-Resamples pro Reihe) wurde auf dem erweiterten öffentlichen
Korpus ausgeführt.

| Kennzahl | Ergebnis |
|---|---:|
| Erfolgreich ausgewertete Reihen | 76 / 76 |
| Signifikant besser als Saisonal-Naiv (5 %) | 74 / 76 |
| Signifikant besser als Persistenz (5 %) | 51 / 76 |
| Medianer Skill vs. Saisonal-Naiv | +23,1 % |
| Medianer Skill vs. Persistenz | +13,9 % |
| Medianer MASE | 0,670 |
| Reihen mit MASE < 1 | 67 / 76 |
| Mittlere P10–P90-Coverage | 80,2 % |
| Coverage-Spanne | 56,7–90,3 % |

Die zwei nicht signifikanten Fälle gegen Saisonal-Naiv sind die stark verwandten
Herne-Reihen `Bezug 10/0,4 kV 2024` und `Bezug 0,4 kV 2024`. Beide erreichen MASE 1,388
und eine deutlich zu niedrige Coverage von rund 63 %.

## Einordnung

- Die 76 Reihen sind nach identischem Werte-Hash dedupliziert, aber **nicht 76 unabhängige
  Stadtwerke**. Der Korpus enthält mehrere Spannungsebenen und mehrere Jahre einzelner Betreiber.
- Die mittlere Coverage trifft das 80-%-Ziel sehr gut, die Streuung zwischen einzelnen Netzen ist
  jedoch zu groß. Vor einem Produktionseinsatz ist eine netzspezifische Online-Kalibrierung nötig.
- Diese Baseline verwendete den globalen Feiertagsparameter `NW`. Die nachfolgende methodische
  Härtung muss das Bundesland je Betreiber führen; die Werte hier bleiben deshalb eine Baseline,
  keine endgültige Produktfreigabe.
- Enedis-/RTE-Regionalreihen sind ein zusätzlicher Robustheitstest und kein Beleg für ein deutsches
  Stadtwerk.
- MAPE wird bei signierten Reihen, Nullwerten und Rückspeisung nicht als Leitmetrik verwendet;
  dort führen Skill, MAE und MASE.

## Datenschutz und Nutzungsrechte

Der Downloader speichert amtlich beziehungsweise regulatorisch veröffentlichte Dateien lokal unter
`data_cache/real/`. Dieser Ordner ist gitignoriert. Veröffentlichungspflicht bedeutet nicht automatisch
eine Erlaubnis zur Weiterverteilung; deshalb enthält GitHub nur Quell-URLs, Prüflogik und abgeleitete
Kennzahlen, niemals die Rohdateien.

## Reproduktion

```powershell
python scripts\fetch_dso_corpus.py --set all --keep-going
python scripts\fetch_fr_public_loads.py
python scripts\build_corpus_index.py
python scripts\build_pool_prior.py --corpus-index data_cache\real\corpus_index.json
python scripts\benchmark_suite.py
python scripts\build_proof_pack.py
```

Lokale Detailartefakte:

- `data_cache/benchmark/benchmark_results.json`
- `data_cache/benchmark/benchmark_table.md`
- `data_cache/benchmark/NetzPilot_Beweis.html`

Sie bleiben wegen der Rohdaten-/Provenienzkette bewusst außerhalb des öffentlichen Git-Repositories.
