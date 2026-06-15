"""Naive Baselines — bleiben in jeder Evaluation enthalten (Pflicht)."""
def persistence(load2d, d):     # gestern, gleiche Stunde
    return load2d[d - 1].copy()
def seasonal_naive(load2d, d):  # Vorwoche, gleiche Stunde
    return load2d[d - 7].copy()
