# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Golden checks for the improved T10 small-utility run."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "data_cache" / "t10_small_utility" / "results.json"


def test_t10_small_utility_meets_proxy_target():
    if not _RESULT.exists():
        return
    with _RESULT.open(encoding="utf-8") as f:
        result = json.load(f)

    runs = result["runs"]
    assert len(runs) >= 2
    r80, r90 = runs[0], runs[1]
    assert r80["metriken"]["model"]["Skill_vs_SaisonalNaiv_%"] >= 10.0
    assert r80["metriken"]["model"]["MAPE_%"] < 13.0
    assert r80["probabilistisch"]["Coverage_Interval_%"] >= 80.0
    assert r90["probabilistisch"]["Coverage_Interval_%"] >= 90.0
    assert "Historical Forecast" in r80["weather_source"]
