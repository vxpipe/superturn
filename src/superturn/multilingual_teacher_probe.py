"""Measure matched-window turn detection on real non-English benchmark turns."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from .ami_calibration import _write_json
from .ami_external_baselines import load_smart_turn
from .audio import RATE, read_audio
from .dense_teacher import SAMPLES, TEACHER_SHA256, file_digest
from .eot_domain_probe import DECISION_MS, GAP, decision_audio, summarize
from .harper_valley_benchmark import STUDENTS
from .model import build
from .progress import Progress
from .sources import data_root

REVISION = "ae834af9e9bddf73d1f550c190aaeb99bf31f523"
DATA = {
    "ar": "02b573fb86f22810c7a903f9359acbaaab5796b7c41dfacc3fbcdc7b895b5848",
    "de": "82c6d47d218107023fecc175c92535b39c8f6d69f5d77715e9e6b27c1b2364e9",
    "es": "68570a61460318463e144db1ff271cb50ea3976aa2cd77e5850b56c6f26fd146",
    "fr": "01514d120d29ddb6b44829c48c5e4bb3e89c77345436ad41b1960f5e96001e36",
    "hi": "1011c399d42583ee8e830545ecf351a289d9c3a96bf1720bf40f9fb08c60a386",
    "id": "ee06b26835c7b04a4e09a93a2024966b939d2cc25fd460cfcff1e6557f5cf0ae",
    "it": "b61ca0b5f92a809b705d27a9ea2343333411a71adac34caf54d74baebcc8ff0d",
    "ja": "518a378abeaaab5736c30750f95a87b8d8d453612639baa0f26373aa11d81eae",
    "ko": "2a083d3b2bca4ccc2c84707f0e1c1078126918c8df54f7d5c2a21e8b1890f002",
    "nl": "b463b06e4437331b65858fce508aa68888e3b4e8168ee0af14e624a7a781c474",
    "pt": "17c2eb6b38b0bb245bb641e20d15c5c710d00b9993b0798d14d611eea67fc12b",
    "tr": "89c613d553b15a665dc8f4d2f882be59bc87e7c3f03af45549004bdad8d8e0ce",
    "zh": "eeb9193e3ac5516ff250f7353a06babef6feefa31a81a1920f85289c4aea073e",
}
WINDOWS = {"smart-turn-3s": SAMPLES, "smart-turn-8s": 8 * RATE}


def pause_spans(row, pcm, language):
    if row["language"] != language or row["cohort"] != "general":
        raise ValueError("unexpected language or cohort among benchmark turns")
    for ordinal, span in enumerate(row["silence_spans"]):
        edge = round(span["start"] * RATE)
        end = min(len(pcm), round(span["end"] * RATE))
        if not 0 <= edge <= end or end - edge < GAP:
            continue
        if span["label"] not in {"eot", "hold"}:
            raise ValueError("unexpected benchmark pause annotation")
        yield ordinal, edge, int(span["label"] == "eot")


def clips_and_labels(rows, language):
    for row in rows:
        pcm = read_audio(row["audio"]["bytes"])
        for _, edge, label in pause_spans(row, pcm, language):
            yield decision_audio(pcm, edge, "telephone"), label


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("multilingual turn benchmarking uses the staged Vast data")
    teacher, teacher_hash = load_smart_turn(root)
    if teacher_hash != TEACHER_SHA256:
        raise ValueError("unexpected Smart Turn checkpoint")
    fits_path = root / "reports" / "dense-teacher-gru.json"
    fits = json.loads(fits_path.read_text())["fit"]
    models = {}
    for name in STUDENTS:
        path = root / "runs" / "dense-teacher-diagnostic" / name / "best.pt"
        if file_digest(path) != fits[name]["checkpoint_sha256"]:
            raise ValueError(f"unexpected student checkpoint: {name}")
        state = torch.load(path, map_location=device, weights_only=True)
        model = build(state["config"]).to(device).eval()
        model.load_state_dict(state["state"])
        models[name] = model
    results = score_languages(root, models, device, teacher=teacher)
    report = {"revision": REVISION, "dataset_sha256": DATA, "teacher_sha256": teacher_hash,
              "student_fit_sha256": file_digest(fits_path), "decision_ms": DECISION_MS,
              "condition": "G.711 telephone", "scope": "provider-labelled diagnostic only",
              "threshold_note": "0.5 descriptive, ROC cutoffs retrospective, no threshold fitted",
              "languages": results}
    _write_json(root / "reports" / "multilingual-teacher-probe.json", report)
    return report


def score_languages(root, models, device, *, teacher=None, paired_rows=None, fixed_thresholds=None, progress_name=None):
    folder = root / "raw" / "livekit_multilingual_diagnostic" / "data"
    sources = {}
    for language, digest in DATA.items():
        path = folder / language / "validation-00000-of-00001.parquet"
        if file_digest(path) != digest:
            raise ValueError(f"{language} benchmark differs from its pinned payload")
        sources[language] = pq.ParquetFile(path)

    def score_teachers(clip):
        return {name: teacher(clip[-length:]) for name, length in WINDOWS.items()}

    results = {}
    step = "multilingual-teacher-probe" if teacher else "depth-multilingual"
    meter = Progress(progress_name or step, sum(p.metadata.num_rows for p in sources.values()), "turns")
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            for language, source in sources.items():
                scores, truth = defaultdict(list), []
                groups, turn_offset = [], 0
                for batch in source.iter_batches(batch_size=8, columns=["audio", "language", "cohort", "silence_spans"]):
                    examples = []
                    for index, row in enumerate(batch.to_pylist()):
                        pauses = list(clips_and_labels((row,), language))
                        examples.extend(pauses)
                        if paired_rows is not None:
                            groups.extend([turn_offset + index] * len(pauses))
                    turn_offset += batch.num_rows
                    if examples:
                        clips, labels = zip(*examples, strict=True)
                        truth.extend(labels)
                        if teacher is not None:
                            for row in workers.map(score_teachers, clips):
                                for name, value in row.items():
                                    scores[name].append(value)
                        x = torch.as_tensor(np.stack([c[-SAMPLES:] for c in clips]), device=device)
                        elapsed = torch.full((len(x), 1), float(DECISION_MS), device=device)
                        with torch.inference_mode():
                            for name, model in models.items():
                                scores[name].extend(model(x, elapsed).flatten().cpu().tolist())
                    meter.advance(batch.num_rows)
                if set(truth) != {0, 1} or any(len(s) != len(truth) or not np.all((0 <= np.asarray(s))
                                                        & (np.asarray(s) <= 1)) for s in scores.values()):
                    raise ValueError(f"{language} has missing labels or invalid scores")
                results[language] = {"turns": source.metadata.num_rows, "holds": len(truth) - sum(truth),
                                     "shifts": sum(truth), "models": {
                                         name: summarize(values, truth, [0.5 if fixed_thresholds is None
                                                                         else fixed_thresholds[name]], roc_limits=True)
                                         for name, values in scores.items()}}
                if paired_rows is not None:
                    paired_rows[language] = {"turn_ordinals": groups, "labels": truth, "scores": dict(scores)}
                if teacher is not None:
                    _write_json(root / "reports" / "multilingual-teacher-progress.json",
                                {"complete_languages": sorted(results), "languages": results})
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return results


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = run(data_root(), "cuda")
    print(json.dumps({lang: rates["turns"] for lang, rates in result["languages"].items()}, indent=2))
