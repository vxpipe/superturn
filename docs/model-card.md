---
license: apache-2.0
tags:
  - onnx
  - turn-taking
  - speech
---

# Superturn

Historical card for the audio-only 973k E20 development artifact. The
active milestone is training a text-only multilingual turn detector.
Release selection and measurements for that model are pending.

## Intended use

Superturn estimates whether a caller has finished their turn at an elapsed
pause. The caller runs its own voice activity detector and supplies a context
window of 16 kHz mono PCM through the decision instant. The combined
candidate also receives the prior turn tail and the current live STT partial.
The release interface will be chosen from the audio-only and combined runs.

| Input | Format |
|---|---|
| `audio` | Float32 `[batch, 48000]` in [-1, 1], zero-padded on the left |
| `silence_ms` | Float32 `[batch, 1]` |
| `text_bytes` (combined only) | Int32 `[batch, L]`, right-aligned normalized UTF-8 with a turn separator, zeros when the current partial is unavailable |
| `p_complete` | Float32 `[batch, 1]` probability output |

The audio-only development artifact uses three seconds of audio, a 512-point
STFT and two 272-wide GRU layers. It has 972,795 learned parameters. The
int8 ONNX file is 998,429 bytes. Independent language thresholds and CPU
latency are pending.
The audio-only budgets are 1 MB int8 and under 2 ms one-core p99.
The combined budgets are 2 MB int8 and at most 3 ms one-core p99.
Report standalone and masked-input results by language.
Fit separate thresholds per supported language on independent calibration
data. Freeze those thresholds before scoring telephone test calls from
unseen speakers. The release requires at least 80% macro balanced accuracy
and at least 70% in each supported language.

## Development evidence at 300 ms

| Candidate | Macro balanced accuracy over 14 languages |
|---|---:|
| Audio-only E20, six float training seeds | 69.97% (69.14–70.78%) |
| Seed 149 float ONNX | 70.75% |
| Seed 149 compact int8 ONNX | 70.52% |
| Project-trained Whisper large-v3 teacher head | 77.24% |
| LiveKit audio Mini | 76.22% |
| Smart Turn v3, 3 s | 68.89% |
| Smart Turn v3, 8 s | 69.65% |

Calibration and evaluation used complementary whole-turn halves.
The benchmark turns lack global person IDs, so speaker-independent
telephone performance remains unverified. Seed 149 was the first seed
in the fixed trial. AUC loss from float ONNX to int8 was at most 0.00145
in each language. The HarperValley-88 comparison is directional because
its calls form one connected caller-agent component.

The combined-input development probe used causal partial transcripts.
On the separate English development benchmark it reached 8.25% shift
recall at a matched 5% false-complete budget, versus 13.50% for its
audio-only control. Its multilingual evaluation requires independent
conversation labels paired with decision-time transcript snapshots.

## Measured operating range

The release will report balanced accuracy, false-complete rate and shift
recall on speaker-disjoint conversation examples at 300 ms. The controlled
telephone sweeps reuse the same examples and frozen language thresholds.

| Telephone condition | To report after evaluation |
|---|---|
| Quiet line | Lowest tested pre-pause speech level in dBFS meeting both targets |
| Fixed -50 dBFS line noise | Lowest tested speech level in dBFS meeting both targets |
| Speech held at -30 dBFS | Lowest tested signal-to-noise ratio in dB meeting both targets |

A reported floor requires meeting both targets at that tested level and all
higher tested levels. If no tested level qualifies, the release will say so.
Speech level is measured from the loudest quarter of 20 ms frames in the last
500 ms before the known pause. The measurements begin at a supplied pause
boundary. Caller audibility and upstream VAD performance require their own
measurement.

## Data and attribution

| Source | Role for the current development artifact | Licence or terms |
|---|---|---|
| Emilia-YODAS, Amphion and the original YODAS2 audio contributors | 18,037 student fit pauses in German, English, French, Japanese, Korean and Chinese | CC BY 4.0. The selected files come from the Emilia-YODAS subset covered by these terms. [Publisher and attribution](https://huggingface.co/datasets/amphion/Emilia-Dataset/blob/main/README.md) |
| VoxPopuli, Meta and the original parliamentary recordings | 595 student fit pauses in Spanish, Italian and Polish | CC0 for the dataset. [Publisher and attribution](https://huggingface.co/datasets/facebook/voxpopuli) |
| HarperValleyBank, Wu, Nafziger, Scodary and Maas | Teacher-head supervision on designated train calls. Separate HarperValley-88 development comparison | CC BY 4.0. [Publisher and attribution](https://github.com/cricketclub/gridspace-stanford-harper-valley) |
| Whisper large-v3, OpenAI | Frozen general-purpose speech encoder for the project-trained teacher | MIT. [Publisher and licence](https://github.com/openai/whisper) |
| LiveKit EoT Benchmark | Development comparisons and owned-teacher model selection using its supplied labels | Dataset-specific terms in the pinned registry. [Publisher and attribution](https://huggingface.co/datasets/livekit/eot-bench-data) |

The student was trained first on intermediate speech features and then
on soft scores from the project-trained head, with a fixed 20-epoch
schedule. The 18,632 student fit pauses have known source-level split
assignments. The student graph contains no Whisper blocks. Reference
turn detectors supplied benchmark scores only.

The dataset registry records the other sources used in earlier experiments,
the corresponding input selections and their attributions. Those runs
include AMI and Common Voice. Their labels and metrics belong to the
individual historical reports. [Dataset options](data-options.md)
