from superturn.paired_call_judgments import compare_continuous, continuous_ballot


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
