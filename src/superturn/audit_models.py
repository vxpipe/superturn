"""Independent neural speech activity for the label audit."""

from .audio import RATE

SPEECHBRAIN_REPO = "speechbrain/vad-crdnn-libriparty"
SPEECHBRAIN_REVISION = "c5d5ae4fce161d94c3ab0286e32fb4a041a21a04"


def speechbrain_detector(root, device):
    import torch
    from huggingface_hub import snapshot_download
    from speechbrain.inference.VAD import VAD

    weights = snapshot_download(SPEECHBRAIN_REPO, revision=SPEECHBRAIN_REVISION,
                                allow_patterns=["*.ckpt", "hyperparams.yaml"])
    detector = VAD.from_hparams(source=weights, savedir=str(root / "audit-models" / "speechbrain-vad"),
                                run_opts={"device": device})
    hop = round(detector.hparams.time_resolution * RATE)

    def detect(pcm, left, right):
        left, right = max(0, left), min(len(pcm), right)
        if right - left < 400:
            return []
        audio = torch.from_numpy(pcm[left:right].astype("float32") / 32768).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = detector.get_speech_prob_chunk(audio).cpu().reshape(-1).tolist()
        segments = []
        start = None
        for index, probability in enumerate(probabilities):
            position = min(right, left + index * hop)
            if probability >= 0.5 and start is None:
                start = position
            elif probability < 0.25 and start is not None:
                if position - start >= 960:
                    segments.append((start, position))
                start = None
        if start is not None and right - start >= 960:
            segments.append((start, right))
        return segments

    return detect
