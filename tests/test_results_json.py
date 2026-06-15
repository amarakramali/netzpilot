"""Guard cached result artifacts against truncated or invalid JSON."""
from pathlib import Path
import json


def test_cached_results_json_files_are_parseable():
    root = Path(__file__).resolve().parents[1]
    cache = root / "data_cache"
    if not cache.exists():
        return

    bad = []
    for path in sorted(cache.rglob("results.json")):
        try:
            with path.open(encoding="utf-8") as f:
                json.load(f)
        except Exception as exc:
            bad.append(f"{path.relative_to(root)}: {exc}")

    assert not bad, "\n".join(bad)
