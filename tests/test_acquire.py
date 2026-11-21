import io
import random
import struct
import tarfile

import pytest

from superturn.acquire import (
    parse_wav_header,
    select_row_groups,
    stream_tar,
    wav_bytes,
    wav_window,
)
from superturn.sources import budget_bytes, load_sources, read_url_list


def _fmt(rate=16_000):
    return struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)


def test_wav_header_round_trip_with_extra_chunks():
    pcm = bytes(range(256)) * 4
    wav = wav_bytes(_fmt(), pcm)
    # Insert a LIST chunk before data, as some recorders do.
    head, data = wav[:36], wav[36:]
    listing = b"LIST" + struct.pack("<I", 4) + b"INFO"
    wav = head[:4] + struct.pack("<I", len(wav) - 8 + len(listing)) + head[8:] + listing + data
    fmt, offset = parse_wav_header(wav)
    assert fmt == _fmt()
    assert wav[offset : offset + len(pcm)] == pcm


def test_parse_wav_header_rejects_non_wav():
    with pytest.raises(ValueError):
        parse_wav_header(b"NIST_1A\n   1024\n")


def test_wav_window_is_shared_by_all_tracks_and_fits_the_shortest():
    one_hour = 44 + 3600 * 32_000
    start, length = wav_window([one_hour, one_hour - 32_000 * 600], 4.0, 0.3)
    assert length == 240.0
    assert start == pytest.approx(0.3 * 3000)
    start, length = wav_window([44 + 60 * 32_000], 4.0, 0.3)  # a 1-minute file
    assert (start, length) == (0.0, 60.0)


def test_select_row_groups_takes_at_least_one_and_respects_the_cap():
    assert select_row_groups([500, 500, 500], 1) == [0]
    assert select_row_groups([100, 100, 100], 250) == [0, 1]
    assert select_row_groups([100, 100], 10_000) == [0, 1]


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_stream_tar_keeps_matching_members_and_stops_at_the_cap(tmp_path):
    rng = random.Random(0)
    files = {f"musan/noise/n{i}.wav": rng.randbytes(50_000) for i in range(10)}
    files["musan/speech/s0.wav"] = b"x" * 10
    archive = _tar_bytes(files)
    result = stream_tar(io.BytesIO(archive), tmp_path, max_bytes=len(archive) // 3, keep=["musan/noise/*"])
    kept = sorted(p.name for p in (tmp_path / "musan" / "noise").iterdir())
    assert 1 <= len(kept) < 10
    assert result["bytes_read"] < len(archive)
    assert not (tmp_path / "musan" / "speech").exists()
    # Kept members are complete.
    for name in kept:
        assert (tmp_path / "musan" / "noise" / name).read_bytes() == files[f"musan/noise/{name}"]


def test_stream_tar_refuses_paths_outside_the_target(tmp_path):
    archive = _tar_bytes({"../escape.txt": b"no"})
    with pytest.raises(RuntimeError):
        stream_tar(io.BytesIO(archive), tmp_path, max_bytes=10**9)


def test_configured_caps_fit_the_budget_without_network():
    """Upper bound from static caps only, so the budget is enforced in review."""
    sources = load_sources()
    bound = 0
    ami = sources["ami"]
    tracks = len(read_url_list(ami))
    bound += tracks * ami.get("minutes_per_meeting") * 60 * 32_000
    for name in ("fleurs", "emilia_yodas"):
        s = sources[name]
        bound += len(s.get("languages")) * s.get("max_bytes_per_file")
    cv = sources["common_voice"]
    bound += len(cv.get("languages")) * (cv.get("max_bytes_per_file") + 5_000_000)  # + TSV
    bound += sources["musan"].get("max_total_bytes")
    bound += 22 * sources["indic_diarbench"].get("max_bytes_per_file")  # 22 language files
    bound += 3 * 600_000_000  # VoxPopuli: one row group (~0.5 GB) for 3 languages
    bound += 200_000_000  # OpenSLR 26
    assert bound <= budget_bytes(), f"{bound / 1e9:.1f} GB"


def test_inspect_archive_lists_member_order_without_writing(tmp_path):
    from superturn.acquire import inspect_archive

    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_tar_bytes({"musan/music/m.wav": b"m" * 1000, "musan/noise/n.wav": b"n" * 10}))
    members = inspect_archive(archive.as_uri(), max_bytes=10**9)
    assert [name for name, _, _ in members] == ["musan/music/m.wav", "musan/noise/n.wav"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.tar.gz"]


def test_request_retries_transient_provider_failure(monkeypatch):
    from urllib.error import HTTPError

    from superturn import acquire

    response = object()
    calls = []
    waits = []

    def open_request(request, *, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 503, "try later", {}, io.BytesIO())
        return response

    monkeypatch.setattr(acquire.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(acquire.time, "sleep", waits.append)
    assert acquire._request("https://example.test/audio") is response
    assert len(calls) == 2 and waits == [1]
