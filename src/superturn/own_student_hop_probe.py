"""Compare 10 and 20 ms feature hops at fixed 973k audio model capacity."""

import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .dense_teacher import SAMPLES, SOURCES, file_digest
from .harper_history_probe import recall_area
from .language_failure_audit import pitch_pair, read_receipt, read_verified
from .model import FRAME, HOP, build
from .multilingual_teacher_probe import DATA, score_languages
from .own_head_distillation import student_scores
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, EPOCHS, SEEDS, compare, fit_arm, read_inputs, targets_for_features
from .sources import data_root
from .spectral import ConvolutionSpectrum
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows

PARAMETERS = 1_000_000
WIDE_HOP = 2 * HOP
RECALL_MARGIN = .01
PAUC_MARGIN = .005
STRATUM_MARGIN = .03


def geometry(hop):
    frames = (SAMPLES - FRAME) // hop + 1
    return frames, (frames + 3) // 4, SAMPLES - ((frames - 1) * hop + FRAME)


def hop_targets(features, valid):
    """Select existing point-interpolated teacher targets at identical centers."""
    def centers(hop):
        return 4 * hop * np.arange(geometry(hop)[1]) + (FRAME - 1) / 2

    original = centers(HOP)[-features.shape[1]:]
    wide = centers(WIDE_HOP)
    desired = wide[wide >= original[0]]
    indices = np.searchsorted(original, desired)
    if not np.array_equal(original[indices], desired) or desired[-1] != original[-1]:
        raise ValueError("teacher values do not share the wider-hop centers and ending")
    if not np.all(valid[:, indices].any(axis=1)):
        raise ValueError("the wide grid lost all valid teacher frames for a pause")
    return features[:, indices], valid[:, indices], desired.tolist()


def frontend_check():
    model = build({"architecture": "a", "window": 3, "parameters": PARAMETERS, "hop": WIDE_HOP}).eval()
    torch.manual_seed(13029)
    pcm = torch.randn((5, SAMPLES)) * .1
    pcm[0].zero_()
    pcm[1, :-4800] = 0
    pcm[2, -4800:] = 0
    pcm[3] *= .001
    with torch.inference_mode():
        power = model.features.power_spectrum(pcm)
        stem = model.stem(model.features(pcm).transpose(1, 2))
        conv = ConvolutionSpectrum(hop=WIDE_HOP)
        torch.testing.assert_close(power, conv.power_spectrum(pcm), rtol=1e-4, atol=2e-4)
        if (power.shape[1], stem.shape[-1]) != geometry(WIDE_HOP)[:2]:
            raise ValueError("incorrect wide-hop frame count")
        gap = torch.full((len(pcm), 1), 300.)
        native = model(pcm, gap)
        model.features = conv
        error = (native - model(pcm, gap)).abs().max().item()
        if error >= 1e-3:
            raise ValueError("the wide-hop convolution frontend changed scores")
    return error


def recall_metrics(values, y):
    areas = np.asarray([recall_area(scores, y) for scores in values]).mean(axis=0)
    return np.asarray([areas[:3].mean(), areas[1], areas[3]])


def summarize(arms, rows):
    rng = np.random.default_rng(29192)
    result, boot, by_arm = {}, [], []
    for language, row in rows.items():
        labels = np.asarray(row["labels"], dtype=bool)
        groups = np.asarray(row["turn_ordinals"])
        scores = [np.asarray([row["scores"][name] for name in names]) for names in arms]
        baseline, wide = [recall_metrics(values, labels) for values in scores]
        paired = wide - baseline
        result[language] = paired.tolist()
        by_arm.append([baseline, wide])
        buckets = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        draws = []
        for sample in rng.integers(len(buckets), size=(400, len(buckets))):
            idx = np.concatenate([buckets[pos] for pos in sample])
            a, b = [recall_metrics(values[:, idx], labels[idx]) for values in scores]
            draws.append(b - a)
        boot.append(draws)
    lower, upper = np.quantile(np.mean(boot, axis=0), [.025, .975], axis=0)
    return {"metrics_order": ["mean_recall_2.5_5_7.5pct", "matched_5pct_recall", "normalized_pauc_10pct"],
            "macro_by_hop": np.asarray(by_arm).mean(axis=0).tolist(), "delta_by_language": result,
            "delta_95": np.column_stack((lower, upper)).tolist(),
            "macro_noninferior": bool(lower[0] > -RECALL_MARGIN and lower[2] > -PAUC_MARGIN
                                      and min(delta[0] for delta in result.values()) >= -STRATUM_MARGIN)}


def english_strata(native, audio, labels, names, scores):
    history = np.array([edge for _, edge, _, _ in native])
    rising = []
    for (pcm, edge, _, _), clip in zip(native, audio, strict=True):
        source = pcm[max(0, edge - (SAMPLES - 4800)):edge].astype(np.float32) / 32768
        source_pitch, _, _ = pitch_pair(source, clip)
        rising.append(source_pitch == "rising")
    result = {}
    truth = np.asarray(labels, dtype=bool)
    for key, subset in (("short_history", history < SAMPLES - 4800), ("rising_pitch", np.asarray(rising))):
        # Use each model's cohort-wide matched 5% threshold, then count subset hits.
        proportions = []
        for group in names:
            recalls = []
            for name in group:
                values = np.asarray(scores[name])
                order = np.argsort(-values, kind="stable")
                ends = np.r_[np.flatnonzero(np.diff(values[order])), len(values) - 1]
                false = np.cumsum(~truth[order])[ends]
                eligible = ends[false <= .05 * (~truth).sum()]
                fired = values >= values[order[eligible[-1]]] if len(eligible) else np.zeros_like(truth)
                recalls.append(fired[subset & truth].mean())
            proportions.append(float(np.mean(recalls)))
        result[key] = {"shifts": int((subset & truth).sum()), "matched_5pct_by_hop": proportions,
                       "delta": proportions[1] - proportions[0]}
    return result


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the hop probe requires staged Vast data")
    reference, reference_hash = read_receipt(root, "own-student-width-probe")
    paired = json.loads(read_verified(root, reference["paired_language_scores"]).read_text())
    cache, digests, pcm, scores, sources, roles, groups, digest = read_inputs(root)
    original, _ = read_receipt(root, "own-teacher-feature-student")
    fit, fidelity = [np.flatnonzero(roles == role) for role in ("fit", "fidelity")]
    if (digest != original["target_report_sha256"] or digest != reference["target_report_sha256"]
            or original["epochs"] != EPOCHS or original["seeds"] != list(SEEDS)
            or len(fit) != original["fit_rows"] or len(fidelity) != original["fidelity_rows"]):
        raise ValueError("the archived training recipe changed")
    pca = root / "runs/own-teacher-feature-student" / digest[:12] / "pca.pt"
    if file_digest(pca) != reference["pca_sha256"]:
        raise ValueError("the fit-only PCA changed")
    parity = frontend_check()
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    features, valid, target_centers = hop_targets(features, valid)
    active = tuple(source for source in SOURCES if source in sources)
    output = root / "runs/own-student-hop-probe" / digest[:12]
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    fits, peers, models = {}, {}, {}
    for seed in SEEDS:
        baseline = model_name(PARAMETERS, seed)
        name = f"hop20-{seed}"
        peers[name] = baseline
        saved = reference["fits"][baseline]
        payload = torch.load(read_verified(root, saved["checkpoint"]), map_location="cpu", weights_only=True)
        fits[baseline] = payload["config"], payload["state"], saved
        fits[name] = fit_arm(pcm, scores, features, valid, sources, groups, fit, fidelity, active,
                             "feature-first", seed, parameters=PARAMETERS, hop=WIDE_HOP)
    metrics, intervals = compare(root, output, fits, peers=peers)
    report = {"scope": "fixed-capacity hop diagnostic", "width_report_sha256": reference_hash,
              "target_report_sha256": digest, "pca_sha256": reference["pca_sha256"],
              "geometry": {"10ms": geometry(HOP), "20ms": geometry(WIDE_HOP)},
              "geometry_fields": ["spectral_frames", "gru_steps", "uncovered_trailing_samples"],
              "aligned_teacher_centers_samples": target_centers, "generated_convolution_score_max_abs": parity,
              "fits": {name: values[2] for name, values in fits.items()}, "metrics": metrics, "paired_english": intervals}
    path = root / "reports/own-student-hop-probe.json"
    _write_json(path, report)
    if any(state is None for _, state, _ in fits.values()):
        return report
    for name in peers:
        config, state, _ = fits[name]
        models[name] = build(config).to(DEVICE).eval()
        models[name].load_state_dict(state)
    rows = {}
    score_languages(root, models, DEVICE, paired_rows=rows, progress_name="own-hop-multilingual")
    if set(rows) != set(DATA) or set(paired) != set(rows) | {"en"}:
        raise ValueError("unexpected development language coverage")
    native = list(english_rows(root))
    clips, truth, groups = zip(*pause_rows(native), strict=True)
    audio = np.stack(clips)
    rows["en"] = {"labels": list(truth), "turn_ordinals": list(groups),
                  "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    for language, row in rows.items():
        past = paired[language]
        if any(row[key] != past[key] for key in ("labels", "turn_ordinals")):
            raise ValueError(f"development membership changed for {language}")
        row["scores"].update({name: past["scores"][name] for name in peers.values()})
    arms = tuple(peers.values()), tuple(peers)
    report["summary"] = summarize(arms, rows)
    report["english_strata"] = english_strata(native, audio, truth, arms, rows["en"]["scores"])
    report["summary"]["english_strata_pass"] = all(cell["delta"] >= -STRATUM_MARGIN
                                                  for cell in report["english_strata"].values())
    pair_path = output / "paired-language-scores.json"
    _write_json(pair_path, rows)
    report["paired_language_scores"] = {"path": pair_path.relative_to(root).as_posix(), "sha256": file_digest(pair_path)}
    _write_json(path, report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root()).get("summary"), indent=2))
