from superturn.paired_call_judgments import compare_continuous, continuous_ballot, window_ballot


def test_continuous_ballots_retain_next_role_disagreement(tmp_path):
    header = ("recording,focal_channel,stop_ms,next_onset_ms,next_channel,"
              "outcome,brief_ack,language,confidence\n")
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text(header + "call-1,left,1000,1400,right,resume,yes,hi,3\n")
    second.write_text(header + "call-1,left,1040,1440,left,resume,no,hindi,3\n"
                      + "call-1,left,2500,3000,right,handoff,no,hindi,2\n")
    key = {"call-1": {"samples": 480_000}}

    counts = compare_continuous(continuous_ballot(first, key), continuous_ballot(second, key))

    assert counts["marks_first"] == 1
    assert counts["marks_second"] == 2
    assert counts["agreed_resume"] == 1
    assert counts["next_role_and_onset_agreement"] == 0


def test_nominal_language_match_requires_agreed_channel_and_common_quiet(tmp_path):
    key = {"call-1": {"samples": 480_000, "nominal_language": "id", "recording": "source-1"}, "call-2": {}}
    forms = tmp_path / "windows.csv"
    forms.write_text("recording,reviewed,left_language,right_language\ncall-1,yes,id,mixed\ncall-2,,,\n")
    reviewed = window_ballot(forms, key)
    assert reviewed == {"call-1": ("indonesian", "mixed")}

    def event(stop, onset, next_channel="right", language="indonesian", outcome="handoff", brief_ack="no"):
        return {("call-1", 0): [(stop, onset, next_channel, {
            "outcome": outcome, "language": language, "brief_ack": brief_ack,
        })]}

    def count(first, second):
        return compare_continuous(first, second, key, (reviewed, reviewed))[
            "joint_nominal_language_and_floor_matches"
        ]

    assert count(event(1000, 1350), event(1010, 1350)) == 1
    assert count(event(1000, 1310), event(1040, 1350)) == 0
    assert count(event(1000, 1350, "left"), event(1010, 1350, "left")) == 0
    assert count(event(1000, 1350, language="tagalog"), event(1010, 1350, language="tagalog")) == 0
    counts = compare_continuous(event(1000, 1350), event(1010, 1350), key, (reviewed, reviewed))
    assert counts["joint_handoff_calls"] == 1
    assert counts["joint_resume_calls"] == 0
    ack = event(1000, 1350, outcome="resume", brief_ack="yes")
    counts = compare_continuous(ack, ack, key, (reviewed, reviewed))
    assert counts["joint_resume_calls"] == 0
    assert counts["acknowledged_partner_first_resumes"] == 1
