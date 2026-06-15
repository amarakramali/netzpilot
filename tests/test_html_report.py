from netzpilot.report.html_report import build_html_report


def test_html_report_contains_provenance_and_coverage():
    summary = {
        "metriken": {
            "model": {
                "MAE_MW": 1.0,
                "RMSE_MW": 2.0,
                "MAPE_%": 3.0,
                "Skill_vs_Persistenz_%": 4.0,
                "Skill_vs_SaisonalNaiv_%": 5.0,
            }
        },
        "probabilistisch": {
            "Interval_Label": "P10-P90",
            "Coverage_Interval_%": 81.0,
            "Coverage_P10_P90_%": 81.0,
            "Soll_%": 80.0,
        },
    }
    html = build_html_report(
        [("demo", summary)],
        {"weather_training_rule": "Historical Forecast API only.", "start_utc_inclusive": "2022-01-01"},
        {"smard_load_hour": {"gap_count": 0, "non_finite_count": 0}},
        ["plots/demo.png"],
    )
    assert "Historical Forecast API only" in html
    assert "81.0% / 80.0%" in html
    assert "plots/demo.png" in html
