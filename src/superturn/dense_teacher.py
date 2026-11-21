"""Prepare real pauses for a matched-window Smart Turn diagnostic."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ami_calibration import _write_json, development_splits
from .ami_distill import TEACHER_SHA256
from .ami_external_baselines import load_smart_turn
from .ami_reference import _manual_spans
from .audio import RATE, read_audio
from .augment import condition
from .labels import Label, ami_speakers, read_rows, turns
from .progress import Progress
from .shards import CONTEXT, SHARD_ROWS, cap_training, clip, seed_for
from .sources import data_root
from .splits import speaker_groups

SOURCES = ("ami", "emilia_yodas", "voxpopuli")
DEADLINES = (200, 300)
SAMPLES = 3 * RATE
AMI_CAP = 48


def ami_candidates(root, observed):
    """Use continuous headset audio and exact annotated word edges."""
    raw = root / "raw"
    archive, = (raw / "ami_annotations").glob("*.zip")
    manifest = json.loads((raw / "ami" / "MANIFEST.json").read_text())
    files = defaultdict(dict)
    for entry in manifest["files"]:
        path = Path(entry["path"])
        channel = int(path.stem.rsplit("Headset-", 1)[1])
        files[path.parts[0]][channel] = entry
    with ZipFile(archive) as metadata:
        roster = ami_speakers(metadata)
        groups = speaker_groups((meeting, (speaker for speaker, _ in tracks.values()))
                                for meeting, tracks in roster.items())
        previous = development_splits({row.recording for row in observed if row.split == "train"}, groups)
        reserved = {groups[row.recording] for row in observed if row.split != "train"}
        reserved.update(groups[m] for m, role in previous.items() if role != "train")
        usable = sorted(m for m in previous if groups[m] not in reserved)
        by_speaker = defaultdict(list)
        meter = Progress("dense-teacher-ami-anchors", len(usable), "meetings")
        try:
            for meeting in usable:
                tracks = roster[meeting]
                (offset, length), = {(round(files[meeting][c]["start_s"] * RATE),
                                      round(files[meeting][c]["length_s"] * RATE)) for c in tracks}
                paths = {speaker: "raw/ami/" + files[meeting][c]["path"]
                         for c, (speaker, _) in tracks.items()}
                spans = list(_manual_spans(metadata, meeting, tracks, offset, length))
                if {span.speaker for span in spans} == paths.keys():
                    for span, gap, tag in turns(spans, decision_ms=DEADLINES[-1]):
                        row = Label("ami", "en", span.speaker, meeting, groups[meeting], "train",
                                    span.end, gap, tag, "human-word-edge", paths[span.speaker])
                        by_speaker[row.speaker].append(row)
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()
    return [row for rows in by_speaker.values() for row in
            sorted(rows, key=lambda r: seed_for("dense-teacher-ami-v1", r.audio, r.pause_start))[:AMI_CAP]], groups


def source_candidates(observed):
    """Connect recording, dataset group and speaker before any role filtering."""
    voices = defaultdict(set)
    for row in observed:
        voices[row.recording].add(("group", row.group))
        if row.speaker and row.speaker != "unknown":
            voices[row.recording].add(("speaker", row.speaker))
    groups = speaker_groups(voices.items())
    reserved = {groups[row.recording] for row in observed if row.split != "train"}
    rows = [row for row in observed if groups[row.recording] not in reserved
            and row.method == "single-track" and row.speaker and row.speaker != "unknown"
            and row.pause_samples >= DEADLINES[-1] * RATE // 1000
            and row.silence_start in (None, row.pause_start)]
    return list(cap_training(rows, 0.005)), groups


def partition(rows, groups):
    source, = {row.source for row in rows}
    keys = {(row.audio, row.pause_start) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"{source} contains duplicate audio pause boundaries")
    candidates = Counter(groups[row.recording] for row in rows)
    if len(candidates) < 5:
        raise ValueError(f"{source} has too few independent groups for fidelity validation")
    count = max(3, round(0.15 * len(candidates)))
    ordered = sorted(candidates, key=lambda group: seed_for("dense-teacher-fidelity-v1", source, group))
    fidelity = set(ordered[:count])
    return [{"role": "fidelity" if groups[row.recording] in fidelity else "fit",
             "connected_group": groups[row.recording], **asdict(row)}
            for row in sorted(rows, key=lambda r: (r.audio, r.pause_start))]


def describe(rows):
    return {"pauses": len(rows), "speakers": len({r["speaker"] for r in rows}),
            "groups": len({r["connected_group"] for r in rows}),
            "languages": dict(Counter(r["language"] for r in rows)),
            "timing_tags": dict(Counter(r["label"] for r in rows))}


def inventory(root, folder):
    selected, summary, inputs = [], {}, {}
    for source in SOURCES:
        path = root / "labels" / f"{source}.jsonl"
        inputs[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        observed = read_rows(path)
        rows, groups = ami_candidates(root, observed) if source == "ami" else source_candidates(observed)
        assigned = partition(rows, groups)
        selected.extend(assigned)
        summary[source] = {role: describe([r for r in assigned if r["role"] == role])
                           for role in ("fit", "fidelity")}
    annotation, = (root / "raw" / "ami_annotations").glob("*.zip")
    for name, path in (("ami_annotations", annotation),
                       ("ami_manifest", root / "raw" / "ami" / "MANIFEST.json")):
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "pool.tmp"
    temporary.write_bytes(encoded)
    temporary.replace(folder / "pool.json")
    report = {"method": "human AMI word edges and natural single-track pauses",
              "use": "diagnostic", "deadlines_ms": DEADLINES, "samples": SAMPLES,
              "pool_sha256": hashlib.sha256(encoded).hexdigest(), "input_sha256": inputs,
              "sources": summary}
    _write_json(root / "reports" / "dense-teacher-inventory.json", report)
    return report


def render(row, pcm, deadline):
    """Cut the future before the G.711 filter and return the shared PCM."""
    end = CONTEXT + deadline * RATE // 1000
    window = clip(row, pcm)
    window[end:] = 0
    return condition(window, "telephone", 0)[end - SAMPLES:end]


def normalize(pcm):
    return pcm.astype(np.float32) / 32768


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def score(root, folder, *, preview=False):
    import _whisper_features

    encoded = (folder / "pool.json").read_bytes()
    inventory_report = json.loads((root / "reports" / "dense-teacher-inventory.json").read_text())
    pool_hash = hashlib.sha256(encoded).hexdigest()
    if pool_hash != inventory_report["pool_sha256"]:
        raise ValueError("pause pool differs from the inventoried selection")
    rows = json.loads(encoded)
    if preview:
        # Cover every source and role while leaving the full selection untouched.
        subsets = (sorted((r for r in rows if r["source"] == source and r["role"] == role),
                          key=lambda r: seed_for("dense-teacher-preview-v1", r["audio"], r["pause_start"]))
                   for source in SOURCES for role in ("fit", "fidelity"))
        rows = sorted((row for subset in subsets for row in subset[:12]),
                      key=lambda r: (r["audio"], r["pause_start"]))
    predict, teacher_hash = load_smart_turn(root)
    if teacher_hash != TEACHER_SHA256:
        raise ValueError("Smart Turn differs from the matched-window evaluation")
    provenance = {"pool_sha256": pool_hash, "teacher_sha256": teacher_hash,
                  "feature_sha256": file_digest(Path(_whisper_features.__file__)),
                  "runner_sha256": file_digest(Path(__file__)), "preview": preview}
    destination = folder / ("preview" if preview else "scored")
    destination.mkdir(exist_ok=True)
    run_file = destination / "inputs.json"
    if run_file.is_file() and json.loads(run_file.read_text()) != provenance:
        raise ValueError("existing scores were made with different inputs")
    _write_json(run_file, provenance)

    @lru_cache(maxsize=8)
    def audio(path):
        return read_audio(root / path)

    def infer(row):
        label = Label(**{name: row[name] for name in Label.__dataclass_fields__})
        pcm = audio(label.audio)
        result = {key: row[key] for key in
                  ("role", "source", "language", "speaker", "connected_group", "recording")}
        for deadline in DEADLINES:
            clip_pcm = render(label, pcm, deadline)
            target = predict(normalize(clip_pcm))
            if not 0 <= target <= 1:
                raise ValueError("teacher returned an invalid probability")
            result[f"pcm_{deadline}"] = clip_pcm.astype("<i2", copy=False).tobytes()
            result[f"target_{deadline}"] = target
        return result

    meter = Progress("dense-teacher-preview" if preview else "dense-teacher-targets",
                     len(rows), "pauses")
    scores = defaultdict(list)
    digests = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            for start in range(0, len(rows), SHARD_ROWS):
                block = rows[start:start + SHARD_ROWS]
                path = destination / f"{start // SHARD_ROWS:04d}.parquet"
                if not path.is_file():
                    table = pa.Table.from_pylist(list(workers.map(infer, block)))
                    tmp = path.with_suffix(".tmp")
                    pq.write_table(table, tmp, compression="zstd")
                    tmp.replace(path)
                values = pq.read_table(path, columns=["source", "role", *(f"target_{t}" for t in DEADLINES)])
                if values.num_rows != len(block):
                    raise ValueError("cached target shard has the wrong row count")
                for row in values.to_pylist():
                    for deadline in DEADLINES:
                        scores[row["source"], row["role"], deadline].append(row[f"target_{deadline}"])
                digests[path.name] = file_digest(path)
                meter.advance(len(block))
        report = {**provenance, "pauses": len(rows), "shard_sha256": digests, "targets": {
            f"{source}/{role}/{deadline}": {
                "pauses": len(values), "quantiles": np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                "over_half": int(np.sum(np.asarray(values) >= 0.5)),
            } for (source, role, deadline), values in sorted(scores.items())}}
        suffix = "preview" if preview else "targets"
        _write_json(root / "reports" / f"dense-teacher-{suffix}.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("inventory", "preview", "score"), nargs="?", default="inventory")
    args = parser.parse_args()
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("dense teacher diagnostics use the audio already on Vast.ai")
    folder = root / "shards" / "dense-teacher"
    if args.step == "inventory":
        report = inventory(root, folder)
    else:
        report = score(root, folder, preview=args.step == "preview")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
