from superturn.yodas_caption_overlap import candidates, grams, tokens


def test_joined_captions_exclude_the_whole_source_video():
    file = ("abcdefghijk-00000-00000000-00000600 One TWO three four.\n"
            "abcdefghijk-00000-00000610-00001210 Five six seven eight!\n"
            "lmnopqrstuv-00000-00000000-00000600 one other caption\n")
    durations = ("abcdefghijk-00000-00000000-00000600 6.0\n"
                 "lmnopqrstuv-00000-00000000-00000600 6.0\n")

    report, blocked = candidates(file.encode(), durations.encode(),
                                 set(grams(tokens("one two three four five six seven eight"))))

    assert report["videos_with_eligible_captioned_duration"] == 2
    assert report["videos_after_text_screen"] == 1
    assert blocked == ["abcdefghijk"]
    assert tokens("हिन्दी") == ("हिन्दी",)
