"""Distil a project-fit speech head using only project-generated soft targets."""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio

from .ami_calibration import _write_json
from .audio import RATE, read_audio
from .dense_teacher import SAMPLES, SOURCES, file_digest, normalize, render
from .dense_teacher_train import SEEDS, fit_model
from .elapsed_turn_probe import paired_intervals
from .eot_domain_probe import load_validated_models
from .harper_ami_probe import predict_head
from .labels import Label
from .model import build
from .multilingual_teacher_probe import score_languages
from .progress import Progress
from .shards import SHARD_ROWS
from .sources import data_root
from .train import _stats, _threshold
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import (
    BUNDLE, CHECKPOINT_SHA256, calibration_rows, extract, pause_rows, rates, speech_features,
)

DEADLINE = 300
BASELINE = "s1-a-150000-29"
PRIMARY_SEED = 29
SCORE_BATCH = 32
MIN_SPREAD = .05


def pool_rows(root):
    folder = root / "shards/dense-teacher"
    report = json.loads((root / "reports/dense-teacher-inventory.json").read_text())
    encoded = (folder / "pool.json").read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != report["pool_sha256"]:
        raise ValueError("the existing pause inventory changed")
    all_rows = json.loads(encoded)
    if {row["source"] for row in all_rows} != set(SOURCES):
        raise ValueError("the inventory contains an unexpected source")
    # The historical AMI cross-validation reserve consumes every AMI
    # fidelity group. Exclude that source for all teachers and students.
    selected = [(i, row) for i, row in enumerate(all_rows) if row["source"] != "ami"]
    for key in ("connected_group", "speaker", "recording"):
        scopes = defaultdict(set)
        for _, row in selected:
            scopes[row["source"], row[key]].add(row["role"])
        if any(len(roles) != 1 for roles in scopes.values()):
            raise ValueError(f"student fit and fidelity overlap by {key}")
    coverage = Counter((row["source"], row["role"]) for _, row in selected)
    expected = {("emilia_yodas", "fit"): 18037, ("emilia_yodas", "fidelity"): 4229,
                ("voxpopuli", "fit"): 595, ("voxpopuli", "fidelity"): 121}
    if coverage != expected:
        raise ValueError("the source and role counts differ from the preflight")
    return folder, digest, selected


def encoder(root, checkpoint):
    path = root / "cache/wav2vec2-harper" / BUNDLE
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    if (checkpoint["encoder_sha256"] != CHECKPOINT_SHA256 or file_digest(path) != CHECKPOINT_SHA256
            or bundle._path != BUNDLE or torchaudio.__version__.split("+")[0] != "2.8.0"):
        raise ValueError("the checkpoint needs the pinned general speech encoder")
    model = torchaudio.models.wav2vec2_model(**bundle._params)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval().to("cuda")


def predictions(features, heads):
    return np.mean([predict_head(features, head) for head in heads], axis=0)


class OwnSpeechHead:
    """Adapt the project's head to the batched diagnostic scoring interface."""

    def __init__(self, model, heads, *, normalization_eps=1e-7):
        self.model = model
        self.heads = heads
        self.normalization_eps = normalization_eps

    def __call__(self, audio, elapsed):
        features = speech_features(self.model, audio, normalization_eps=self.normalization_eps).cpu().numpy()
        return torch.as_tensor(predictions(features, self.heads), device=audio.device)


def render_and_score(root, folder, rows, model, heads):
    pcm = np.empty((len(rows), 1, SAMPLES), dtype="<i2")
    targets = {arm: np.empty((len(rows), 1), dtype=np.float32) for arm in heads}
    @lru_cache(maxsize=6)
    def audio(path):
        return read_audio(root / path)

    @lru_cache(maxsize=6)
    def prior(shard):
        # Read only previously rendered PCM for the decoder parity check.
        path = folder / "scored" / f"{shard:04d}.parquet"
        return pq.read_table(path, columns=["pcm_300"]).column("pcm_300")

    parity = Counter()
    meter = Progress("own-head-targets-300ms", len(rows), "pauses")
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), SCORE_BATCH):
                stop = min(start + SCORE_BATCH, len(rows))
                for out, (index, row) in enumerate(rows[start:stop], start):
                    label = Label(**{key: row[key] for key in Label.__dataclass_fields__})
                    original = audio(label.audio)
                    samples = render(label, original, DEADLINE).astype("<i2", copy=False)
                    key = row["source"], row["role"]
                    if parity[key] < 3:
                        old_bytes = prior(index // SHARD_ROWS)[index % SHARD_ROWS].as_py()
                        if samples.tobytes() != old_bytes:
                            raise ValueError("the re-rendered PCM differs from the previous decoder")
                        changed = original.copy()
                        deadline = label.pause_start + DEADLINE * RATE // 1000
                        if deadline < len(changed):
                            changed[deadline:deadline + RATE // 10] ^= 32767
                            if not np.array_equal(samples, render(label, changed, DEADLINE)):
                                raise ValueError("samples after the decision affected the causal input")
                            parity[key] += 1
                    pcm[out, 0] = samples
                batch = torch.as_tensor(normalize(pcm[start:stop, 0]), device="cuda")
                features = speech_features(model, batch).cpu().numpy()
                for arm, ensemble in heads.items():
                    targets[arm][start:stop, 0] = predictions(features, ensemble)
                meter.advance(stop - start)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    if set(parity.values()) != {3}:
        raise ValueError("not enough nonempty future perturbations for each source and role")
    if any(not np.all((values >= 0) & (values <= 1)) for values in targets.values()):
        raise ValueError("the project head returned invalid probabilities")
    return pcm, targets, {"/".join(key): count for key, count in parity.items()}


def validate_teachers(root, model, heads, provenance, cohorts):
    clips = np.stack([clip for rows in cohorts.values() for clip, _, _ in rows]).astype(np.float32)
    # Extract in the same original batch geometry, including caller rows,
    # so exact serialized head replay is checked before bulk scoring.
    from .wav2vec_harper_probe import training_rows

    training = np.stack([clip for clip, _, _ in training_rows(root)])
    combined = np.concatenate((training, clips))
    features = extract("own-head-replay", lambda audio: speech_features(model, audio), combined)
    if hashlib.sha256(features.tobytes()).hexdigest() != provenance["original_features_sha256"]:
        raise ValueError("the serialized head replay pooled features differ")
    labels = np.asarray([truth for rows in cohorts.values() for _, truth, _ in rows])
    reports = {}
    n_english, n_harper = (len(cohorts[key]) for key in ("english", "harper"))
    stop = n_english + n_harper
    for arm, ensemble in heads.items():
        p = predictions(features[len(training):], ensemble)
        if hashlib.sha256(p.tobytes()).hexdigest() != provenance["own_head_probability_sha256"][arm]:
            raise ValueError(f"the archived {arm} head predictions did not reproduce")
        threshold = _threshold(p[stop:], labels[stop:])
        reports[arm] = {"ami_threshold": threshold}
        for cohort, idx in (("english", slice(0, n_english)), ("harper", slice(n_english, stop))):
            reports[arm][cohort] = {**rates(p[idx], labels[idx]),
                                    "ami_fixed": _stats(p[idx].tolist(), labels[idx].tolist(), threshold)}
    by_language = score_languages(root, {arm: OwnSpeechHead(model, head) for arm, head in heads.items()}, "cuda")
    return reports, by_language


def student_scores(model, clips, device="cuda"):
    scores = []
    with torch.inference_mode():
        for start in range(0, len(clips), SCORE_BATCH):
            x = torch.as_tensor(clips[start:start + SCORE_BATCH], device=device)
            elapsed = torch.full((len(x), 1), float(DEADLINE), device=device)
            scores.append(model(x, elapsed).flatten().cpu().numpy())
    return np.concatenate(scores)


def evaluate_students(root, fits, cohorts):
    clips = np.stack([clip for rows in cohorts.values() for clip, _, _ in rows]).astype(np.float32)
    labels = np.asarray([truth for rows in cohorts.values() for _, truth, _ in rows])
    [(name, baseline, baseline_threshold)], provenance = load_validated_models(root, [BASELINE])
    if name != BASELINE or baseline.samples != SAMPLES:
        raise ValueError("the development control differs from the archived 145k GRU")
    scores = {"direct-label-baseline": student_scores(baseline, clips, device="cpu")}
    ensembles = defaultdict(list)
    for (arm, seed, shuffled), (path, report) in fits.items():
        if file_digest(path) != report["checkpoint_sha256"]:
            raise ValueError("a fitted student checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build(saved["config"]).eval().to("cuda")
        model.load_state_dict(saved["state"])
        key = f"{arm}-shuffle" if shuffled else arm
        score = student_scores(model, clips)
        ensembles[key].append(score)
        scores[key if seed == PRIMARY_SEED else f"{key}-seed-{seed}"] = score
    scores.update({f"{arm}-two-model-mean": np.mean(values, axis=0) for arm, values in ensembles.items()})
    n_english, n_harper = (len(cohorts[key]) for key in ("english", "harper"))
    stop = n_english + n_harper
    metrics = {"english": {}, "harper": {}}
    for arm, p in scores.items():
        threshold = baseline_threshold if arm == "direct-label-baseline" else _threshold(p[stop:], labels[stop:])
        for cohort, idx in (("english", slice(0, n_english)), ("harper", slice(n_english, stop))):
            metrics[cohort][arm] = {**rates(p[idx], labels[idx]), "ami_threshold": threshold,
                                     "ami_fixed": _stats(p[idx].tolist(), labels[idx].tolist(), threshold)}
    compared = {arm: "direct-label-baseline" if arm == "caller" else "caller"
                for arm in ensembles if not arm.endswith("-shuffle")}
    paired = paired_intervals({arm: p[:n_english] for arm, p in scores.items()}, labels[:n_english],
                              [row[2] for row in cohorts["english"]], compared, partial_auc=True)
    reference = metrics["english"]["direct-label-baseline"]["recalls_at_2_5_5_7_5pct"][1]
    if not np.isclose(reference, .135, rtol=0, atol=1e-12):
        raise ValueError("the archived 13.5% development baseline did not reproduce")
    return metrics, paired, provenance


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the distillation experiment uses only staged Vast audio")
    reports = root / "reports"
    transfer_report = reports / "harper-ami-transfer.json"
    provenance = json.loads(transfer_report.read_text())
    head_path = root / provenance["head_checkpoint"]["path"]
    if file_digest(head_path) != provenance["head_checkpoint"]["sha256"]:
        raise ValueError("the project head checkpoint differs from the fitted head report")
    payload = torch.load(head_path, map_location="cpu", weights_only=True)
    if payload["ensemble"] != "mean-probability" or set(payload["arm_states"]) != set(provenance["own_head_probability_sha256"]):
        raise ValueError("unexpected project head ensemble")
    heads = payload["arm_states"]
    model = encoder(root, payload)
    cohorts = {"english": list(pause_rows(english_rows(root))),
               "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
               "calibration": list(calibration_rows(root))}
    teacher, languages = validate_teachers(root, model, heads, provenance, cohorts)
    folder, pool_hash, rows = pool_rows(root)
    _write_json(reports / "own-speech-teacher.json", {
        "pool_sha256": pool_hash, "head_sha256": file_digest(head_path), "decision_ms": DEADLINE,
        "english_and_harper": teacher, "multilingual_diagnostic": languages,
    })
    pcm, soft_targets, parity = render_and_score(root, folder, rows, model, heads)
    del model
    torch.use_deterministic_algorithms(True)
    sources, roles, groups, language = (np.asarray([row[key] for _, row in rows])
                                        for key in ("source", "role", "connected_group", "language"))
    eligible_hash = hashlib.sha256(json.dumps([i for i, _ in rows]).encode()).hexdigest()
    target_report = {"pool_sha256": pool_hash, "head_sha256": file_digest(head_path), "decision_ms": DEADLINE,
                     "eligible_ordinals_sha256": eligible_hash,
                     "pauses": len(rows), "decoder": {"pyav": av.__version__, "libraries": av.library_versions},
                     "pcm_sha256": hashlib.sha256(memoryview(pcm)).hexdigest(), "prefix_and_decoder_checks": parity,
                     "source_roles": dict(Counter(f"{s}/{r}" for s, r in zip(sources, roles, strict=True))),
                     "arms": {}}
    output = root / "runs/own-head-distillation"
    output.mkdir(parents=True, exist_ok=True)
    for arm, values in soft_targets.items():
        target_file = output / f"{arm}-targets.npy"
        target_tmp = target_file.with_suffix(".tmp")
        with target_tmp.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        target_tmp.replace(target_file)
        target_report["arms"][arm] = {
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "file": {"path": target_file.relative_to(root).as_posix(), "sha256": file_digest(target_file)},
            "source_language_quantiles": {f"{s}/{l}": np.quantile(values[(sources == s) & (language == l)],
                                                               [.01, .1, .5, .9, .99]).tolist()
                                          for s, l in sorted(set(zip(sources, language, strict=True)))},
        }
    target_path = reports / "own-speech-targets.json"
    _write_json(target_path, target_report)
    target_hash = file_digest(target_path)
    fits = {}
    for arm, targets in soft_targets.items():
        arm_hash = hashlib.sha256(json.dumps([target_hash, arm, target_report["arms"][arm]["sha256"]]).encode()).hexdigest()
        for seed in SEEDS:
            for shuffled in (False, True):
                name = f"{arm}-{seed}-{'shuffle' if shuffled else 'true'}"
                config = {"architecture": "a", "window": 3, "parameters": 150000,
                          "seed": seed, "shuffled": shuffled}
                fits[arm, seed, shuffled] = fit_model(output / name, config, pcm, targets, sources, roles, groups,
                                                     arm_hash, "cuda", deadlines=(DEADLINE,))
    del pcm, soft_targets
    metrics, intervals, control = evaluate_students(root, fits, cohorts)
    recall = {arm: m["recalls_at_2_5_5_7_5pct"][1] for arm, m in metrics["english"].items()}
    gates = {}
    for arm in heads:
        ref = "direct-label-baseline" if arm == "caller" else "caller"
        fidelity = fits[arm, PRIMARY_SEED, False][1]["selected"]["sources"]
        shuffled_fidelity = fits[arm, PRIMARY_SEED, True][1]["selected"]["sources"]
        rank = {s: value["by_deadline"][str(DEADLINE)]["spearman"] for s, value in fidelity.items()}
        null_rank = {s: value["by_deadline"][str(DEADLINE)]["spearman"] for s, value in shuffled_fidelity.items()}
        gates[arm] = {
            "recall_gain": recall[arm] + 1e-12 >= recall[ref] + (.05 if arm == "caller" else .03),
            "positive_paired_lower_bound": intervals[arm]["recall_delta_95"][0] > 0,
            "exceeds_shuffle": recall[arm] > recall[f"{arm}-shuffle"],
            "harper_partial_auc_direction": metrics["harper"][arm]["partial_auc"] >= metrics["harper"][ref]["partial_auc"],
            "not_collapsed": all(source["by_deadline"][str(DEADLINE)]["score_spread"] >= MIN_SPREAD
                                 for source in fidelity.values()),
            "fidelity_exceeds_shuffle": all(rank[s] is not None and rank[s] > 0
                                            and (null_rank[s] is None or rank[s] > null_rank[s]) for s in rank),
        }
    result = {"target_report_sha256": target_hash, "control": control, "decision_ms": DEADLINE,
              "primary_seed": PRIMARY_SEED,
              "thresholds": "archived threshold for the baseline; 51 AMI holds for students",
              "teacher": teacher, "metrics": metrics, "paired_english": intervals,
              "fits": {f"{arm}/{seed}/{shuffled}": report for (arm, seed, shuffled), (_, report) in fits.items()},
              "development_gates": gates}
    _write_json(reports / "own-head-distillation.json", result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root())["development_gates"], indent=2))
