"""Leakage-sichere Feature-Erzeugung fuer Day-ahead (Stichtag 00:00 lokal).

Modelliert die Wochenabweichung r(d,h) = load(d,h) - load(d-7,h) ("Saisonal-Naiv + Korrektur").
Es werden NUR Informationen verwendet, die am Stichtag d 00:00 bekannt sind:
  - Lasten bis Tag d-1 (Lags, Vortagesstatistik),
  - Kalender/Feiertage von Tag d,
  - Wetter-FORECAST fuer Tag d (sofern uebergeben; Forecast, nicht Istwert!).
"""
from __future__ import annotations
import numpy as np, pandas as pd

def to_daily(series: pd.Series):
    """Contiguous stuendliche Series -> (load2d[ND,24], days[DatetimeIndex lokal]).
    Validiert Stundenraster; lokalisiert nach Europe/Berlin fuer Kalenderfeatures."""
    s = series.sort_index()
    idx = s.index.tz_convert("Europe/Berlin") if s.index.tz else s.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    diffs = s.index.to_series().diff().dropna()
    if not diffs.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Stundenraster nicht lueckenlos — vor Feature-Bau bereinigen (DST/Luecken).")
    n = len(s) - (len(s) % 24)
    vals = s.values[:n].astype(float)
    ND = n // 24
    load2d = vals.reshape(ND, 24)
    days = pd.to_datetime([idx[d * 24].date() for d in range(ND)])
    return load2d, days

def weather_to_daily(weather: pd.DataFrame, days) -> np.ndarray:
    """Convert an hourly forecast-weather frame aligned to load hours into [ND, 24, F]."""
    w = weather.sort_index()
    idx = w.index.tz_convert("Europe/Berlin") if w.index.tz else w.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    diffs = w.index.to_series().diff().dropna()
    if not diffs.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Weather forecast hourly grid is not contiguous.")
    n = len(days) * 24
    if len(w) < n:
        raise ValueError("Weather forecast frame is shorter than load frame.")
    vals = w.iloc[:n].to_numpy(dtype=float)
    weather_days = pd.to_datetime([idx[d * 24].date() for d in range(len(days))])
    if not weather_days.equals(pd.DatetimeIndex(days)):
        raise ValueError("Weather forecast days do not align with load days.")
    return vals.reshape(len(days), 24, vals.shape[1])

def get_holidays(years, region="NW"):
    try:
        import holidays as _h
        return set(_h.Germany(years=list(years), subdiv=region).keys())
    except Exception:
        # Fallback ohne Paket: nur bundesweite Fixtermine (genuegt fuer Sandbox/Smoke).
        out = set()
        for y in years:
            out |= {pd.Timestamp(f"{y}-01-01").date(), pd.Timestamp(f"{y}-05-01").date(),
                    pd.Timestamp(f"{y}-10-03").date(), pd.Timestamp(f"{y}-12-25").date(),
                    pd.Timestamp(f"{y}-12-26").date()}
        return out


def apply_holiday_overrides(holiday_set, add_dates=None, remove_dates=None):
    """Nutzer-kuratierte Kalender-Korrekturen — explizit statt geraten.

    add_dates:    Tage, die WIE EIN FEIERTAG behandelt werden sollen (z. B. Brückentag,
                  lokaler Feiertag, Betriebsferien eines dominanten Industriekunden).
    remove_dates: Tage, die NICHT als Feiertag behandelt werden sollen (Kalender-Korrektur).

    Die Daten fließen in DENSELBEN holiday_set, den die gemessene Maschinerie nutzt
    (Feiertags-Merkmal des Zieltags, feiertagsbewusster Vorwochen-Anker, Trainings-Targets) —
    kein neuer, unvermessener Sonderpfad. Ehrliche Einordnung: „wie Feiertag behandeln" ist
    eine NUTZER-Annahme; gemessen belegt ist das Verhalten echter Feiertage.

    Akzeptiert ISO-Strings ("2026-05-15"), date- oder Timestamp-Objekte. Ungültige Daten
    -> ValueError (kein stilles Verschlucken).
    """
    def _to_date(x):
        try:
            return pd.Timestamp(x).date()
        except Exception:
            raise ValueError(f"Ungültiges Datum in Feiertags-Override: {x!r} (erwartet ISO, z. B. 2026-05-15)")
    out = set(holiday_set or set())
    for x in (add_dates or []):
        out.add(_to_date(x))
    for x in (remove_dates or []):
        out.discard(_to_date(x))
    return out


def base(load2d, d):            # Saisonal-Naiv-Basis
    return load2d[d - 7].copy()

def resid_target(load2d, d):    # Zielgroesse des Korrekturmodells
    return load2d[d] - load2d[d - 7]

def build_features(load2d, days, d, weather2d=None, holiday_set=None):
    """Feature-Matrix [H, F] fuer Zieltag d (leakage-sicher)."""
    H = int(load2d.shape[1])
    dev_prev = load2d[d - 1] - load2d[d - 8]       # gestern vs. Vorwoche-gestern
    dev_mean = float(dev_prev.mean())
    trend = float(load2d[d - 1].mean() - load2d[d - 8].mean())
    dow = days[d].dayofweek
    wknd = 1.0 if dow >= 5 else 0.0
    hol = 1.0 if (holiday_set and days[d].date() in holiday_set) else 0.0
    X = []
    for h in range(H):
        row = [1.0,
               dev_prev[h], dev_mean, trend,
               load2d[d - 1, h] - load2d[d - 7, h],
               np.sin(2*np.pi*h/H), np.cos(2*np.pi*h/H),
               np.sin(4*np.pi*h/H), np.cos(4*np.pi*h/H),
               np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
               wknd, hol]
        if weather2d is not None:                  # Wetter-FORECAST fuer Tag d (T3)
            row += list(weather2d[d, h])
        X.append(row)
    return np.array(X, dtype=float)


def build_small_load_features(load2d, days, d, weather2d=None, holiday_set=None):
    """Feature matrix for volatile small-utility loads.

    Extends the national-load feature set with strictly past short-term
    deviations and simple morning/evening interactions. All added lags are known
    at the day-ahead forecast cut-off.
    """
    if d < 14:
        raise ValueError("build_small_load_features requires at least 14 prior days")
    X = build_features(load2d, days, d, weather2d, holiday_set)
    dow = days[d].dayofweek
    wknd = 1.0 if dow >= 5 else 0.0
    day_delta_mean = float(load2d[d - 1].mean() - load2d[d - 8].mean())
    extra = []
    H = int(load2d.shape[1])
    for h in range(H):
        hour = h * 24.0 / H
        lag2 = load2d[d - 2, h] - load2d[d - 9, h]
        lag3 = load2d[d - 3, h] - load2d[d - 10, h]
        lag14 = load2d[d - 7, h] - load2d[d - 14, h] if d >= 14 else 0.0
        roll3 = float(np.mean([load2d[d - i, h] for i in (1, 2, 3)])) - load2d[d - 7, h]
        roll7 = float(np.mean([load2d[d - i, h] for i in range(1, 8)])) - load2d[d - 7, h]
        evening = float(np.exp(-((hour - 19) ** 2) / 8.0))
        morning = float(np.exp(-((hour - 8) ** 2) / 8.0))
        extra.append([
            lag2,
            lag3,
            lag14,
            roll3,
            roll7,
            day_delta_mean * evening,
            day_delta_mean * morning,
            wknd * evening,
            wknd * morning,
        ])
    return np.hstack([X, np.asarray(extra, dtype=float)])


# --- T3: DST-robuste Lokaltag-Aufbereitung (Mehrjahresdaten mit Sommerzeit) ---
def _complete_local_day(g, hourcol):
    hrs = sorted(g[hourcol].tolist())
    return len(g) == 24 and hrs == list(range(24))


def to_daily_local(series, tz="Europe/Berlin"):
    """Stuendliche UTC-Series -> (load2d[ND,24], days, good_dates) in LOKALER Zeit.
    Behaelt nur vollstaendige Lokaltage; verwirft die ~2 DST-Umstelltage/Jahr (23h/25h),
    damit die Stunde-im-Tag konsistent zur Ortszeit bleibt (kein UTC-Reshape-Versatz)."""
    import numpy as _np, pandas as _pd
    s = series.sort_index()
    loc = s.index.tz_convert(tz) if s.index.tz else s.index.tz_localize("UTC").tz_convert(tz)
    df = _pd.DataFrame({"v": _np.asarray(s.values, float)}, index=loc)
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    groups = {}
    for date, g in df.groupby("date"):
        if _complete_local_day(g, "hour"):
            groups[date] = g.sort_values("hour")["v"].to_numpy()
    good = sorted(groups)
    load2d = _np.array([groups[d] for d in good])
    days = _pd.to_datetime([d.date() for d in good])
    return load2d, days, good


def frame_to_daily_local(frame, good_dates, tz="Europe/Berlin"):
    """Forecast-Wetter-Frame (stuendlich, UTC) -> [ND,24,F], ausgerichtet auf good_dates."""
    import numpy as _np, pandas as _pd
    f = frame.sort_index().copy()
    loc = f.index.tz_convert(tz) if f.index.tz else f.index.tz_localize("UTC").tz_convert(tz)
    f.index = loc
    cols = list(frame.columns)
    f["__date"] = f.index.normalize()
    f["__hour"] = f.index.hour
    by = {}
    for d, g in f.groupby("__date"):
        if _complete_local_day(g, "__hour"):
            by[d] = g.sort_values("__hour")[cols].to_numpy(float)
    return _np.array([by[d] for d in good_dates])
