import json
from pathlib import Path

from scripts.dataset_manifest import MANIFEST


def test_t28_corpus_index_counts_and_filters():
    path = Path("data_cache/real/corpus_index.json")
    if not path.exists():
        return
    corpus = json.loads(path.read_text(encoding="utf-8"))
    assert corpus["n_series"] >= 30
    assert 20 <= corpus["n_independent_networks"] <= corpus["n_series"]
    assert corpus["n_correlation_redundant"] == corpus["n_series"] - corpus["n_independent_networks"]
    assert corpus["n_pool_series"] >= 30
    assert corpus["n_duplicates_excluded"] >= 1
    assert corpus["independence_method"]["correlation_threshold"] == 0.98
    assert corpus["country_counts"]["FR"] >= 3
    assert corpus["network_kind_counts"]["tso_regional"] >= 1
    assert len(corpus["entries"]) == corpus["n_entries_total"]
    bad = [e for e in corpus["entries"] if not e["full_year_like"]]
    assert bad
    assert all(not e["include_in_pool"] for e in bad)
    assert all(e["n_hours"] >= 7_200 for e in corpus["entries"] if e["include_in_benchmark"])
    assert all(e.get("country") in {"DE", "FR"} for e in corpus["entries"])
    assert all(e.get("network_kind") in {"dso_real", "tso_regional"} for e in corpus["entries"])
    hashes = [e["value_hash"] for e in corpus["entries"] if e["include_in_benchmark"]]
    assert len(hashes) == len(set(hashes))
    corr_redundant = [e for e in corpus["entries"] if e.get("redundant_of")]
    assert corr_redundant
    assert all(e["independent_network"] is False for e in corr_redundant)
    assert sum(1 for e in corpus["entries"] if e.get("independent_network")) == corpus["n_independent_networks"]


def test_t28_pool_prior_is_real_corpus_normalized():
    path = Path("data_cache/pool/pool_prior.json")
    if not path.exists():
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    assert prior["n_houses"] >= 30
    assert "last-normalisiert" in prior["space"]
    assert prior["n_features"] == len(prior["w_pool"])


def test_dataset_manifest_uses_valid_corpus_entries():
    assert len(MANIFEST) >= 30
    assert all(e["csv"].startswith("data_cache/real/") for e in MANIFEST)
    assert any(e.get("country") == "FR" and e.get("network_kind") == "dso_real" for e in MANIFEST)
    assert any(e.get("country") == "FR" and e.get("network_kind") == "tso_regional" for e in MANIFEST)
    hashes = [e.get("value_hash") for e in MANIFEST]
    assert len(hashes) == len(set(hashes))
    assert sum(1 for e in MANIFEST if e.get("independent_network")) >= 20


def test_t28c_network_independence_report_exists():
    path = Path("data_cache/benchmark/network_independence.md")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    assert "Network Independence" in text
    assert "High-Correlation Edges" in text
