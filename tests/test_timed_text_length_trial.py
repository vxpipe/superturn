from superturn.timed_text_length_trial import report_path


def test_training_and_ranking_reports_have_distinct_paths(tmp_path):
    assert report_path(tmp_path, score_only=False) == (
        tmp_path / "reports/publisher-timed-english-length-trial.json"
    )
    assert report_path(tmp_path, score_only=True) == (
        tmp_path / "reports/publisher-timed-english-length-trial-ranking.json"
    )
