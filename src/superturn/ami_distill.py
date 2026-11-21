"""Test whether Smart Turn probabilities improve the audio-only AMI GRU."""

from functools import lru_cache
from pathlib import Path

from .ami_deadline import inputs
from .ami_external_baselines import load_smart_turn
from .ami_word_supervision import run_supervision
from .audio import RATE, read_audio
from .sources import data_root

TEACHER_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
DEADLINES = (200, 300)


def run(root, device):
    import _whisper_features

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI distillation uses the training server")
    predict, checkpoint_sha256 = load_smart_turn(root)
    if checkpoint_sha256 != TEACHER_SHA256:
        raise ValueError("Smart Turn checkpoint differs from the evaluated reference")
    assets = {
        "teacher_onnx": root / "cache" / "eot-competitors" / "smart-turn-v3.2-cpu.onnx",
        "teacher_features": Path(_whisper_features.__file__),
        "teacher_adapter": Path(__file__).with_name("ami_external_baselines.py"),
        "teacher_target_source": Path(__file__),
        "deadline_render": Path(__file__).with_name("ami_deadline.py"),
    }
    cached_audio = lru_cache(maxsize=6)(read_audio)
    cached_targets = {}

    def make_targets(rows):
        missing = [row for row in rows if (row["audio"], row["pause_start"]) not in cached_targets]
        if missing:
            rendered, _ = inputs(root, missing, 8 * RATE, cached_audio)
            for i, row in enumerate(missing):
                targets = [predict(rendered[gap][i]) for gap in DEADLINES]
                if not all(0 <= score <= 1 for score in targets):
                    raise ValueError("Smart Turn returned an invalid probability")
                cached_targets[row["audio"], row["pause_start"]] = targets
        return [cached_targets[row["audio"], row["pause_start"]] for row in rows]

    return run_supervision(root, device, "ami-smart-turn-distillation", "smart-turn-targets", make_targets, assets)


if __name__ == "__main__":
    print(run(data_root(), "cuda"))
