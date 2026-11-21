# Local turn pipeline

## 2026-11-21

- AMI's provider HEAD returned 22,887,865 bytes for manual annotations v1.6.2.
  `bin/plan-downloads` reported 18.1 GB total and
  `bin/plan-downloads --sample` reported 0.9 GB. Planning read provider
  metadata. No corpus payload was fetched locally.
- Added per-track AMI labels from the speaker map and VAD, with a dialogue
  act backchannel check. Indic uses human timestamps. Whisper tiny aligns
  known read-speech text, and cuts use trailing silence from the same
  recording. VoxPopuli and Emilia-YODAS retain a distinct weak-shift label.
- Shards retain eight seconds of context and up to three seconds of silence.
  The speaker cap is 0.5 percent of the original training candidate count per
  source. FLEURS does not expose global speaker IDs and stays test-only.
  The shard manifest records speaker counts and a deterministic holdout hash.
- Added seeded augmentation, a recurrent candidate and a conformer-style
  candidate. ONNX export uses STFT after the first explicit Fourier matrix
  graph exceeded 1 MB. In the exported graph, linear layers use dynamic
  quantization. GRU and convolution weights are stored as int8 and
  dequantized for their runtime kernels. The pitch check initially missed
  100 Hz by 6.7 percent. Compensating for the analysis window passed the
  6 percent check on 100, 160, 230 and 340 Hz reference tones.
- `uv run --offline --locked pytest -q` reported 31 passed and four Torch-dependent skips. The full suite
  passed all 35 tests with the existing Python 3.14 CPU Torch environment. The 1 s recurrent fixture had 144,975 parameters and a
  300,447-byte artifact. The conformer fixture had 154,016 parameters and a
  388,961-byte artifact. Float parity probes had maximum absolute error
  0.0 for both. The fixture benchmark used three timing samples and serves
  as a pipeline check. Campaign latency and quality gates remain open.
- Local wheels totalled roughly 160 MB based on package sizes. The separate
  CPU fixture environment reused the existing Torch install. The Python 3.12
  project lock pins Torch 2.8.0 and the rest of the CUDA campaign stack,
  which is installed on Vast.ai.
- `bin/campaign-smoke` isolates sample input and exposes its progress and
  reports in the campaign root. `bin/vast-offers` now prices the configured
  200 GiB and uses the CLI's result limit. No machine was rented. First live
  smoke work needs rental approval.
