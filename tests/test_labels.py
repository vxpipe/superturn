from array import array
from itertools import groupby
import json
import wave
from zipfile import ZipFile

from superturn.labels import Speech, ami_labels, turns


def speech(speaker, start, end):
    return Speech(speaker, start * 16, end * 16)


def test_turns_classify_holds_shifts_and_overlap():
    assert list(turns([speech("a", 0, 1000), speech("b", 1100, 2200)])) == []
    assert list(turns([speech("a", 0, 1000), speech("b", 900, 2200)])) == []
    a = speech("a", 0, 1000)
    b = speech("b", 1200, 2500)
    assert list(turns([b, a])) == [(a, 3200, "shift")]
    assert list(turns([a, speech("a", 1350, 2000)])) == [(a, 5600, "hold")]
    assert list(turns([a, speech("b", 1200, 1400), speech("a", 1800, 2300)]))[0] == (a, 3200, "hold")


def test_ami_maps_channels_to_global_speakers_and_flags_backchannel(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERTURN_DATA", str(tmp_path))
    raw = tmp_path / "raw"
    ami = raw / "ami"
    ami.mkdir(parents=True)
    annotations = raw / "ami_annotations"
    annotations.mkdir()
    with ZipFile(annotations / "annotations.zip", "w") as archive:
        archive.writestr("corpusResources/meetings.xml", """
        <meetings><meeting observation="ES2002a">
        <speaker channel="0" global_name="P1" nxt_agent="A"/>
        <speaker channel="1" global_name="P2" nxt_agent="B"/>
        </meeting><meeting observation="ES2001a">
        <speaker channel="0" global_name="P1" nxt_agent="A"/>
        </meeting></meetings>""")
        archive.writestr("words/ES2002a.B.words.xml",
                         '<words><w id="word1" starttime="11.2"/></words>')
        archive.writestr("dialogueActs/ES2002a.B.dialog-act.xml", """
        <acts><act><pointer href="da-types.xml#id(ami_da_1)"/>
        <child href="ES2002a.B.words.xml#id(word1)"/></act></acts>""")
    (ami / "MANIFEST.json").write_text(json.dumps({"files": [
        {"path": f"ES2002a/audio/ES2002a.Headset-{channel}.wav", "start_s": 10}
        for channel in range(2)
    ]}))
    for channel, activity in enumerate(([(0, 16000), (28800, 36800)], [(19200, 22400)])):
        path = ami / f"ES2002a/audio/ES2002a.Headset-{channel}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = array("h", [0]) * 40000
        for start, end in activity:
            pcm[start:end] = array("h", [4096]) * (end - start)
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm.tobytes())

    def fixture(path):
        with wave.open(str(path)) as wav:
            pcm = array("h", wav.readframes(wav.getnframes()))
        spans = []
        offset = 0
        for voiced, samples in groupby(pcm, key=bool):
            end = offset + sum(1 for _ in samples)
            if voiced:
                spans.append((offset, end))
            offset = end
        return spans

    first, *_ = list(ami_labels(tmp_path, fixture))
    assert (first.speaker, first.group, first.label) == ("P1", "ES2001a", "hold")
    assert first.pause_samples == 3200 and first.annotated_backchannel
