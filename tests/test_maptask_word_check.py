from superturn.labels import Speech
from superturn.maptask_word_check import check


def test_uncertain_intervening_word_prevents_silence_confirmation():
    caller = Speech("0", 0, 16000)
    partner = Speech("1", 24000, 42000)
    words = [(caller, "there"), (partner, "start")]
    clear, _ = check([(caller, partner, "shift")], words)
    uncertain, _ = check([(caller, partner, "shift")], words, [Speech("0", 17000, 19000)])
    assert clear["by_class"]["shift"] == {"supported": 1}
    assert uncertain["by_class"]["shift"] == {"uncertain_intervening_word": 1}


def test_word_check_does_not_skip_an_earlier_turn_to_find_a_match():
    caller = Speech("0", 0, 16000)
    partner = Speech("1", 24000, 42000)
    earlier = Speech("0", 18000, 19000)
    result, _ = check([(caller, partner, "shift")], [(caller, "there"), (earlier, "wait"), (partner, "start")])
    assert result["by_class"]["shift"] == {"no_word_end": 1}
