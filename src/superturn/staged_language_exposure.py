"""Seal causal pause inputs for the staged six-language exposure comparison."""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .audio import RATE, read_audio, silero
from .augment import condition
from .balanced_development_scorecard import UNSEEN_IN_FIT
from .dense_teacher import SAMPLES, file_digest, source_candidates
from .labels import read_rows
from .own_head_distillation import pool_rows
from .progress import Progress
from .shards import seed_for
from .sources import data_root

JOB = "staged-language-exposure-v2"
QUIET = round(.3 * RATE)
PER_CLIP = 2
PER_CALL = 12


def candidate_gaps(spans, length):
    """Yield exact acoustic edges with at least 300 ms of recorded following quiet."""
    for index, (start, edge) in enumerate(spans):
        interior = index + 1 < len(spans)
        stop = spans[index + 1][0] if interior else length
        if edge - start >= RATE // 5 and stop - edge >= QUIET:
            yield edge, "inside" if interior else "tail"


def clip_pcm(pcm, edge, key):
    deadline = edge + QUIET
    start = deadline - SAMPLES
    observed = pcm[max(start, 0):deadline]
    if deadline > len(pcm):
        raise ValueError("no recorded tail to support this decision")
    missing = SAMPLES - len(observed)
    if missing:
        noise = np.random.default_rng(seed_for(JOB, key, edge)).normal(0, 32768 * 10 ** (-55 / 20), missing)
        observed = np.concatenate((np.rint(noise).astype("<i2"), observed))
    return condition(observed, "telephone", 0), max(0, -start)


def distinct_paths(path, source, forbidden=frozenset(), fidelity=()):
    rows = list(read_rows(path))
    protected_voices = {row.speaker for row in rows if row.split != "train"}
    protected_groups = {row.group for row in rows if row.split != "train"}
    if fidelity:
        _, connected = source_candidates(rows)
        protected = {row["connected_group"] for row in fidelity}
        forbidden = forbidden | {row.audio for row in rows if connected[row.recording] in protected}
    eligible = {row.audio: row for row in rows if row.split == "train" and row.audio not in forbidden
                and row.speaker not in protected_voices | {"unknown", ""}
                and row.group not in protected_groups
                and (source != "common_voice" or row.language in UNSEEN_IN_FIT)}
    return sorted(eligible.values(), key=lambda row: seed_for(JOB, source, row.audio))


def examples(pcm, spans, *, language, speaker, clip, source, cap=PER_CLIP, partner=()):
    possible = [(edge, kind) for edge, kind in candidate_gaps(spans, len(pcm))
                if not any(a < edge + QUIET and b > max(0, edge + QUIET - SAMPLES) for a, b in partner)]
    chosen = sorted(possible, key=lambda item: seed_for(JOB, clip, item[0]))[:cap]
    rows = []
    for edge, kind in chosen:
        audio, missing = clip_pcm(pcm, edge, clip)
        row = {"source": source, "language": language, "speaker": str(speaker), "clip": clip,
               "edge_sample": edge, "gap_kind": kind, "missing_history_samples": missing,
               "pcm_sha256": hashlib.sha256(audio.astype("<i2").tobytes()).hexdigest()}
        rows.append((row, audio))
    return rows


def single_files(root, source, rows, detector):
    meter = Progress(f"{JOB}-{source}", len(rows), "recordings")
    result = []
    clips, offers = Counter(), Counter()
    try:
        for row in rows:
            speaker = row.language, row.speaker
            if source == "common_voice" and clips[speaker] >= 2:
                meter.advance(1)
                continue
            pcm = read_audio(root / row.audio)
            found = examples(pcm, detector(pcm), language=row.language, speaker=row.speaker,
                             clip=row.audio, source=source)
            if found:
                clips[speaker] += 1
            for item in found:
                meta = item[0]
                geometry = speaker, meta["gap_kind"], bool(meta["missing_history_samples"])
                if source != "extra_emilia" or offers[geometry] < 2:
                    result.append(item)
                    offers[geometry] += 1
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return result


def dutch(root, detector, protected):
    folder = root / "raw/voxpopuli-dutch-exposure"
    receipt = json.loads((folder / "MANIFEST.json").read_text())
    path = folder / "train-row-group-6.parquet"
    if file_digest(path) != receipt["local_parquet_sha256"]:
        raise ValueError("Dutch audio selection changed")
    table = pq.read_table(path)
    result = []
    meter = Progress(f"{JOB}-dutch", len(table), "recordings")
    try:
        for row in table.to_pylist():
            if row["speaker_id"] not in protected:
                pcm = read_audio(row["audio"]["bytes"])
                result.extend(examples(pcm, detector(pcm), language="nl", speaker=row["speaker_id"],
                                       clip=row["audio_id"], source="voxpopuli_dutch"))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return result


def paired_calls(root, detector):
    from .datahive_conversation_pilot import inventory, load_call
    from .dral_label_pilot import isolation
    from .spring_hindi_clock import load_tracks, read_inputs

    result, rejected = [], Counter()

    def collect(tracks, name, source, languages):
        activity = [detector(channel) for channel in tracks]
        if not isolation(tracks, activity)["passed"]:
            rejected[f"{source}_isolation"] += 1
            return
        for side, (pcm, language) in enumerate(zip(tracks, languages, strict=True)):
            if language is None:
                rejected[f"{source}_language"] += 1
                continue
            # Participant identity across calls is unresolved. Use the entire
            # recording as one conservative weighting group.
            result.extend(examples(pcm, activity[side], language=language, speaker=name,
                                   clip=f"{name}:{side}", source=source, cap=PER_CALL // 2,
                                   partner=activity[1 - side]))

    plan, audio, folder, _, _ = read_inputs(root)
    for selected in plan["selection"]:
        name = selected["recording"]
        collect(load_tracks(folder, audio, selected), name, "spring", ("hi", "hi"))
    source, folder, _, metadata = inventory(root)
    archive = json.loads((root / "reports/datahive-conversation-pilot.json").read_text())["calls"]
    for name, digest in sorted(source.get("audio_sha256").items()):
        desc = archive[name]
        nominal = desc["language"]
        languages = []
        for blocks in desc["language_posteriors_by_channel"]:
            wins = sum(bool(block) and block[0][0] == nominal and block[0][1] >= .60 for block in blocks)
            languages.append(nominal if nominal in ("hi", "id") and blocks and wins * 2 >= len(blocks) else None)
        if not any(languages):
            rejected["datahive_no_language_supported"] += 1
            continue
        tracks, actual = load_call(folder, name, digest, metadata)
        if any(actual[key] != desc[key] for key in actual):
            raise ValueError("the DataHive PCM differs from its staged language scan")
        collect(tracks, name, "datahive", languages)
    return result, rejected


def matching_control(exposure, offers):
    """Match the count by true acoustic gap kind and presence of padded history."""
    def geometry(row):
        return row["gap_kind"], bool(row["missing_history_samples"])

    strata, controls = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for item in exposure:
        strata[geometry(item[0])][item[0]["language"]].append(item)
    for item in offers:
        controls[geometry(item[0])].append(item)
    for by_language in strata.values():
        for items in by_language.values():
            items.sort(key=lambda item: seed_for(JOB, item[0]["clip"], item[0]["edge_sample"]))
    selected, control = [], []
    coverage = Counter()
    control_voices = Counter()
    language_cells = Counter(lang for by_language in strata.values() for lang in by_language)
    for group, by_language in sorted(strata.items(), key=lambda pair: len(controls[pair[0]])):
        positions = Counter()
        available = controls[group]
        for offered in available:
            speaker = offered[0]["speaker"]
            if control_voices[speaker] == 2:
                continue
            languages = [lang for lang, items in by_language.items() if positions[lang] < len(items)]
            if not languages:
                break
            lang = min(languages, key=lambda lang: (coverage[lang], language_cells[lang], lang))
            selected.append(by_language[lang][positions[lang]])
            control.append(offered)
            control_voices[speaker] += 1
            coverage[lang] += 1
            positions[lang] += 1
    counts = lambda rows: Counter((row["language"], *geometry(row)) for row, _ in rows)
    before, after = counts(exposure), counts(selected)
    audit = {"exposure_by_cell": [
        {"language": lang, "geometry": kind, "padded": padded, "eligible": count,
         "matched": after[lang, kind, padded]} for (lang, kind, padded), count in sorted(before.items())],
        "control_supply": [{"geometry": kind, "padded": padded, "eligible": len(rows)}
                           for (kind, padded), rows in sorted(controls.items())]}
    return selected, control, audit


def archive(root, name, selected):
    directory = root / "runs" / JOB
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    audio = directory / f"{name}.npy"
    meta = [row for row, _ in selected]
    if path.exists() or audio.exists():
        if not path.exists() or not audio.exists() or json.loads(path.read_text()) != meta:
            raise ValueError("the sealed exposure replay changed")
        persisted = np.load(audio, allow_pickle=False)
        rows_match = persisted.shape == (len(meta), SAMPLES) and all(
            hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest() == row["pcm_sha256"]
            for pcm, row in zip(persisted, meta, strict=True))
        if not rows_match:
            raise ValueError("the sealed exposure PCM changed")
    else:
        with audio.with_suffix(".partial").open("wb") as stream:
            np.save(stream, np.stack([pcm for _, pcm in selected]), allow_pickle=False)
        audio.with_suffix(".partial").replace(audio)
        _write_json(path, meta)
    counts = Counter((row["source"], row["language"], row["gap_kind"]) for row in meta)
    people = defaultdict(set)
    for row in meta:
        people[row["source"], row["language"]].add(row["speaker"])
    return {"rows": len(meta), "membership_sha256": file_digest(path), "audio_sha256": file_digest(audio),
            "coverage": [{"source": source, "language": language, "gap": gap, "rows": count}
                         for (source, language, gap), count in sorted(counts.items())],
            "speaker_or_call_groups": [{"source": source, "language": language, "groups": len(keys)}
                                       for (source, language), keys in sorted(people.items())]}


def run(root):
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("staged recordings stay on Vast")
    torch.set_num_threads(2)
    detector = silero()
    _, pool_digest, rows = pool_rows(root)
    fidelity = defaultdict(list)
    for _, row in rows:
        if row["role"] == "fidelity":
            fidelity[row["source"]].append(row)
    cv = distinct_paths(root / "labels/common_voice.jsonl", "common_voice")
    b = single_files(root, "common_voice", cv, detector)
    b.extend(dutch(root, detector, {row["speaker"] for row in fidelity["voxpopuli"]}))
    phones, rejected = paired_calls(root, detector)
    b.extend(phones)
    used = {row["audio"] for _, row in rows}
    candidates = distinct_paths(root / "labels/emilia_yodas.jsonl", "emilia_yodas", used, fidelity["emilia_yodas"])
    b, c, matching = matching_control(b, single_files(root, "extra_emilia", candidates, detector))
    missing = sorted(set(UNSEEN_IN_FIT) - {row["language"] for row, _ in b})
    report = {"scope": "sealed fit-only causal inputs with no labels or teacher targets",
              "paired_call_scope": "training feasibility recordings only; never independent evaluation",
              "input_data_download_bytes": 0, "base_pool_sha256": pool_digest,
              "missing_languages": missing, "matching": matching,
              "exposure": archive(root, "exposure", b) if b else None,
              "control": archive(root, "extra-emilia", c) if c else None,
              "channel_filter_counts": dict(rejected)}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
