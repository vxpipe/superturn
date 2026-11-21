---
license: apache-2.0
tags:
  - onnx
  - turn-taking
  - speech
---

# Superturn

Draft for `HashNuke/superturn`. Candidate selection and release measurements
are pending.

## Intended use

Superturn estimates whether a caller has finished their turn at an elapsed
pause. The caller runs its own voice activity detector and supplies a context
window of 16 kHz mono PCM through the decision instant and the current live
partial transcript if available. This is the approved combined interface
for the release candidate.

| Input | Format |
|---|---|
| `audio` | Float32 `[batch, 48000]` in [-1, 1], zero-padded on the left |
| `silence_ms` | Float32 `[batch, 1]` |
| `text_bytes` | Int32 `[batch, 128]`, right-aligned normalized UTF-8, zeros if unavailable |
| `p_complete` | Float32 `[batch, 1]` probability output |

The release will record the selected context length, parameter count,
ONNX file size, probability threshold and single-core CPU latency.
The int8 budget is 2 MB total and the one-core p99 target is at most 3 ms
for both branches. Report masked audio and masked text results by language.
Final thresholds are fitted on English AMI telephone validation pauses.
The benchmark applies that threshold across languages and audio conditions.

## Measured operating range

The release will report the false-complete rate and complete recall on
speaker-disjoint conversation examples at 300 ms. The controlled telephone
sweeps reuse the same examples and the same validation-selected threshold.

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

| Dataset | Role | Licence |
|---|---|---|
| AMI Meeting Corpus and manual annotations | English conversation training and speaker-disjoint evaluation | CC BY 4.0 |
| Mozilla Common Voice 17.0 | Read-speech training | CC0 |
| VoxPopuli | Spontaneous speech training using the dataset | CC0 |
| Emilia-YODAS | Spontaneous speech training using the YODAS subset | CC BY 4.0 |
| FLEURS | Read-speech evaluation | CC BY 4.0 |
| Indic DiarBench | Non-English conversation evaluation | CC BY 4.0 |
| MUSAN Free Sound | Environmental noise augmentation | MUSAN corpus CC BY 4.0, selected recordings in the US public domain |
| OpenSLR 26 | Simulated room impulse responses | Apache 2.0 |

The dataset registry records upstream providers, exact selections and
attributions. Training sources are stratified by language and grouped by
speaker. The test reports provide separate source, language and condition
metrics with case counts. FLEURS provides utterance IDs and lacks global
speaker IDs in the metadata used here. [Dataset options](data-options.md)

The automated label review did not meet the original 90% target on the
fixed AMI sample. Training uses 522 AMI pauses from 132 speakers after
annotation checks and speaker capping. The median speaker contributes
four pauses and the most represented speaker contributes 12. The next annotated speech must
start at least 200 ms after the proposed pause boundary, and training
queries stop at that onset. Validation and test use the original
automatically generated labels. Published metrics therefore include
evaluation labels without manual confirmation.
An additional AMI benchmark reports false completes and recall at 300 ms
on 55 English test pauses from 10 speakers. The annotations support the
speaker order and put the next speech onset at or after the 300 ms query.
The measured pause begins at the Silero boundary and the time match
allows for differences between that boundary and annotated speech end.
The benchmark uses the same validation-selected threshold as the full test.
AMI's transcripts and dialogue-act tags were produced by people. AMI
obtained the word timestamps by automatically aligning those transcripts
to the audio. [Transcription process](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml) [Dialogue-act annotations](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml)
