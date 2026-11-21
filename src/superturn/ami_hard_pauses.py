"""Reconstruct bounded AMI training-only audio for candidate hard pauses."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import struct
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .acquire import _request, parse_wav_header, wav_bytes
from .ami_calibration import _write_json
from .ami_pilot import GAP, acoustic_status, annotated_pauses, group_words
from .ami_reference import _manual_spans
from .audio import RATE, read_audio
from .dense_teacher import file_digest
from .labels import Label, Speech, ami_backchannels, ami_speakers
from .progress import Progress
from .shards import seed_for
from .sources import REPO_ROOT, budget_bytes, data_root, load_sources, read_url_list
from .wav2vec_harper_probe import verified_ami_labels

SOURCE = "ami-hard-pauses"
TILE = 30 * RATE
TILES = 24
MIN_SPEECH = RATE


def observed_history(spans, index):
    caller = spans[index]
    starts = [caller.start]
    for previous in reversed(spans[:index]):
        if previous.speaker != caller.speaker:
            if previous.end <= caller.start and previous.end - previous.start >= MIN_SPEECH:
                return caller.end - min(starts)
        elif previous.end <= caller.start:
            starts.append(previous.start)
    return None


def make_plan(root, archive_path):
    source = load_sources()["ami"]
    index_path = REPO_ROOT / source.get("list")
    prefix = source.get("strip_prefix")
    files = defaultdict(dict)
    for size, url in read_url_list(source):
        relative = url.removeprefix(prefix)
        meeting = relative.split("/")[0]
        channel = int(relative.rsplit("Headset-", 1)[1].split(".")[0])
        files[meeting][channel] = {"url": url, "size": size}
    original_path = root / "raw/ami/MANIFEST.json"
    original = {}
    for asset in json.loads(original_path.read_text())["files"]:
        meeting = asset["path"].split("/")[0]
        offset = round(asset["start_s"] * RATE)
        original[meeting] = offset, offset + round(asset["length_s"] * RATE)
    fit = verified_ami_labels(root, "train")
    fit_meetings = {row.recording: row.group for row in fit}
    protected = [row for role in ("validation", "calibration", "assessment", "crosscheck")
                 for row in verified_ami_labels(root, role)]
    if ({row.group for row in fit} & {row.group for row in protected}
            or {row.speaker for row in fit} & {row.speaker for row in protected}):
        raise ValueError("AMI training and evaluation identities overlap")
    protected_voices = {row.speaker for row in protected}
    tiles = {}
    with ZipFile(archive_path) as archive:
        people = ami_speakers(archive)
        meter = Progress("ami-hard-plan", len(fit_meetings), "meetings")
        for meeting, group in sorted(fit_meetings.items()):
            speakers = people[meeting]
            if {speaker for speaker, _ in speakers.values()} & protected_voices:
                raise ValueError("a selected recording contains an evaluation speaker")
            tracks = {speaker: files[meeting][channel] for channel, (speaker, _) in speakers.items()}
            duration = min((track["size"] - 4096) // 2 for track in tracks.values())
            spans = group_words(_manual_spans(archive, meeting, speakers, 0, duration))
            order = {span: index for index, span in enumerate(spans)}
            backchannels = {s: ami_backchannels(archive, meeting, agent, 0) for s, agent in speakers.values()}
            old_left, old_right = original[meeting]
            for caller, following, label, backchannel in annotated_pauses(spans, backchannels):
                if backchannel:
                    continue
                start = caller.end // TILE * TILE
                if start + TILE > duration or start < old_right and start + TILE > old_left:
                    continue
                if caller.end < start + 3 * RATE or following.start + RATE // 4 > start + TILE:
                    continue
                key = meeting, start
                candidate = tiles.setdefault(key, {"recording": meeting, "group": group, "start": start,
                                             "frames": TILE, "tracks": tracks, "pauses": []})
                candidate["pauses"].append({"caller": asdict(caller), "following": asdict(following), "label": label,
                                             "prior_turn_samples": observed_history(spans, order[caller])})
            meter.advance(1)
        meter.finish()
    coverage, meetings, groups, selection = Counter(), Counter(), Counter(), []

    def order_key(item):
        key, tile = item
        short = Counter(row["label"] for row in tile["pauses"]
                        if row["prior_turn_samples"] is not None and row["prior_turn_samples"] < 2.7 * RATE)
        return -min(short["hold"], short["shift"]), -sum(short.values()), seed_for("ami-hard-pauses-v1", *key)

    for key, tile in sorted(tiles.items(), key=order_key):
        if meetings[tile["recording"]] or groups[tile["group"]] >= 4:
            continue
        selection.append(tile)
        meetings[tile["recording"]] += 1
        groups[tile["group"]] += 1
        for pause in tile["pauses"]:
            kind = f"{pause['label']}/short" if pause["prior_turn_samples"] is not None and (
                pause["prior_turn_samples"] < 2.7 * RATE) else f"{pause['label']}/other"
            coverage[kind] += 1
        if len(selection) == TILES:
            break
    if not selection:
        raise ValueError("no unseen training-only AMI ranges qualify")
    return {"version": 1, "source": SOURCE, "annotation_sha256": file_digest(archive_path),
            "original_audio_manifest_sha256": file_digest(original_path), "url_list_sha256": file_digest(index_path),
            "short_history_definition": "mapped caller speech history since last clearly sustained partner bout",
            "planned_audio_bytes": sum(len(tile["tracks"]) * TILE * 2 for tile in selection),
            "candidate_counts": coverage, "training_meeting_groups": len(groups), "tiles": selection}


def exact_range(url, total, start, end):
    with _request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as response:
        if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("AMI mirror did not honor its pinned range")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("AMI mirror returned an unexpected payload length")
    return data


def get_track(root, tile, speaker, item, pinned):
    if not speaker or "/" in speaker or "\\" in speaker or speaker in (".", ".."):
        raise ValueError("invalid AMI speaker identifier")
    relative = f"{tile['recording']}/{tile['start']}/{speaker}.wav"
    path = root / "raw" / SOURCE / relative
    expected = pinned.get(relative)
    if path.is_file() and expected is not None:
        if path.stat().st_size != expected["bytes"] or file_digest(path) != expected["sha256"]:
            raise ValueError("previously staged AMI audio disagrees with its reconstruction receipt")
        return path
    header = exact_range(item["url"], item["size"], 0, 4095)
    fmt, begin = parse_wav_header(header)
    if struct.unpack("<HHIIHH", fmt[:16]) != (1, 1, RATE, RATE * 2, 2, 16):
        raise ValueError("AMI headset is not native mono 16 kHz PCM")
    start = begin + 2 * tile["start"]
    samples = exact_range(item["url"], item["size"], start, start + 2 * tile["frames"] - 1)
    payload = wav_bytes(fmt, samples)
    if expected is not None and (len(payload), hashlib.sha256(payload).hexdigest()) != (
            expected["bytes"], expected["sha256"]):
        raise ValueError("AMI mirror sample content changed since this dataset was locked")
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError("previously staged source bytes changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return path


def stage(root, plan, archive_path, receipt):
    folder = root / "raw" / SOURCE
    manifest_path = folder / "MANIFEST.json"
    manifests = (manifest_path, root / "manifests" / f"{SOURCE}.json")
    pinned_receipts = [json.loads(path.read_text()) for path in manifests if path.is_file()]
    if pinned_receipts and any(record["selection_sha256"] != file_digest(receipt)
                               or record != pinned_receipts[0] for record in pinned_receipts):
        raise ValueError("AMI reconstruction receipts disagree")
    pinned = {row["path"]: row for row in pinned_receipts[0]["files"]} if pinned_receipts else {}
    raw_bytes = sum(path.stat().st_size for path in (root / "raw").rglob("*") if path.is_file())
    reserve = sum(len(tile["tracks"]) * (2 * tile["frames"] + 4096 + 64) for tile in plan["tiles"])
    if reserve > load_sources()[SOURCE].get("max_total_bytes") or raw_bytes + reserve > budget_bytes():
        raise ValueError("planned AMI ranges exceed the campaign's raw data budget")
    meter = Progress("ami-hard-pauses", len(plan["tiles"]), "training windows")
    labels, fetches, flags = [], [], Counter()
    with ZipFile(archive_path) as archive, ThreadPoolExecutor(max_workers=8) as workers:
        people = ami_speakers(archive)
        try:
            for tile in plan["tiles"]:
                meeting = tile["recording"]
                assets = list(sorted(tile["tracks"].items()))
                jobs = [workers.submit(get_track, root, tile, speaker, item, pinned) for speaker, item in assets]
                tracks = {speaker: job.result() for (speaker, _), job in zip(assets, jobs, strict=True)}
                audio = {speaker: read_audio(path) for speaker, path in tracks.items()}
                for path in tracks.values():
                    fetches.append({"path": path.relative_to(folder).as_posix(),
                                    "sha256": file_digest(path), "bytes": path.stat().st_size})
                start = tile["start"]
                nonwords = list(_manual_spans(archive, meeting, people[meeting], start, tile["frames"],
                                              tags=("vocalsound", "disfmarker", "gap", "transformerror")))
                for row in tile["pauses"]:
                    caller, following = (Speech(item["speaker"], max(0, item["start"] - start),
                                                item["end"] - start)
                                         for item in (row["caller"], row["following"]))
                    status = acoustic_status(caller, following, audio, nonwords)
                    flags[status] += 1
                    if status != "passed":
                        continue
                    label = Label(SOURCE, "en", caller.speaker, meeting, tile["group"], "train",
                                  caller.end, GAP, row["label"], "human-annotation-acoustic",
                                  tracks[caller.speaker].relative_to(root).as_posix())
                    labels.append({"pause_id": f"ami:{meeting}:{caller.speaker}:{caller.end + start}",
                                   "label": asdict(label), "prior_turn_samples": row["prior_turn_samples"]})
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
    meter.finish()
    manifest = {"source": SOURCE, "selection_sha256": file_digest(receipt), "files": fetches}
    for prior in manifests:
        if prior.is_file() and json.loads(prior.read_text()) != manifest:
            raise ValueError("reconstructed AMI audio hashes differ from the existing receipt")
    _write_json(manifest_path, manifest)
    output = root / "labels/ami-hard-pauses.json"
    selected = {"plan_sha256": file_digest(receipt), "manifest_sha256": file_digest(manifest_path),
                "pauses": labels}
    if output.is_file() and json.loads(output.read_text()) != selected:
        raise ValueError("reconstructed AMI pause IDs differ from the existing receipt")
    _write_json(output, selected)
    report = {"plan_sha256": file_digest(receipt), "recordings": len(plan["tiles"]),
              "verified": dict(Counter(row["label"]["label"] for row in labels)),
              "short": dict(Counter(row["label"]["label"] for row in labels
                                    if row["prior_turn_samples"] is not None and row["prior_turn_samples"] < 2.7 * RATE)),
              "speakers": len({row["label"]["speaker"] for row in labels}), "screen": flags,
              "label_manifest_sha256": file_digest(output)}
    _write_json(root / "reports/ami-hard-pauses.json", report)
    return report


def verify_rebuild(root, plan, archive_path, receipt):
    staged = root / "raw" / SOURCE / "MANIFEST.json"
    labels = root / "labels/ami-hard-pauses.json"
    with TemporaryDirectory(prefix="ami-hard-rebuild-", dir=root) as path:
        scratch = Path(path)
        for name in ("raw", "manifests", "labels", "reports"):
            (scratch / name).mkdir()
        copyfile(staged, scratch / "manifests" / f"{SOURCE}.json")
        copyfile(labels, scratch / "labels" / labels.name)
        rebuilt = stage(scratch, plan, archive_path, receipt)
        if (file_digest(scratch / "raw" / SOURCE / "MANIFEST.json") != file_digest(staged)
                or rebuilt["label_manifest_sha256"] != file_digest(labels)):
            raise ValueError("fresh reconstruction changed the staged audio or labels")
    rebuilt["fresh_download_verified"] = True
    _write_json(root / "reports/ami-hard-pauses-rebuild-verification.json", rebuilt)
    return rebuilt


def run(root, *, plan_only=False, verify_only=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI source reconstruction requires the campaign machine")
    archive_path, = (root / "raw/ami_annotations").glob("*.zip")
    receipt = root / "reports/ami-hard-pauses-plan.json"
    candidate = make_plan(root, archive_path)
    if receipt.is_file():
        if json.loads(receipt.read_text()) != candidate:
            raise ValueError("AMI planning inputs changed; an existing membership is pinned")
    else:
        _write_json(receipt, candidate)
    if plan_only:
        return {key: item for key, item in candidate.items() if key != "tiles"}
    if verify_only:
        return verify_rebuild(root, candidate, archive_path, receipt)
    return stage(root, candidate, archive_path, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--verify-rebuild", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(data_root(), plan_only=args.plan_only, verify_only=args.verify_rebuild), indent=2))
