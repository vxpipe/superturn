from superturn.text_cpu_probe import encoder_screen


def test_encoder_screen_requires_language_coverage_and_unclipped_token_lengths():
    detail = {"token_needs": {"en": {"full_two_lines": {"n": 10, "p90": 51.0}},
                              "kn": {"full_two_lines": {"n": 0}}}}
    durations = {str(n): {"synthetic_text_to_scalar": {"p99_ms": 4.9}} for n in (64, 128)}
    load = {"workers_with_cpu_ticks": 2}

    def screen():
        return encoder_screen(detail, ("en", "kn"), (64, 128), durations, durations, load,
                              expected_workers=2)

    missing = screen()
    assert not missing["passed"]
    assert missing["missing_languages"] == ["kn"]

    detail["token_needs"]["kn"]["full_two_lines"] = {"n": 10, "p90": 128.1}
    over_limit = screen()
    assert not over_limit["passed"]
    assert over_limit["p90_exceeds_128"]
    assert over_limit["conservative_measured_bucket"] is None

    detail["token_needs"]["kn"]["full_two_lines"]["p90"] = 63.1
    within_limit = screen()
    assert within_limit["passed"]
    assert within_limit["conservative_measured_bucket"] == 64
