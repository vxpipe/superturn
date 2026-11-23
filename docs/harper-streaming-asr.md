# HarperValley streaming ASR condition

Run `bin/harper-streaming-capture --limit-per-class 2` as a Vast campaign job
for a smoke check. Run the same command without the limit for the frozen
English development pack. This condition leaves benchmark membership fixed
and records availability separately for each outcome class.

The English Zipformer checkpoint is pinned to revision
`672fbf1b30579d6585301139bb363f42a0ad4a24` and its publisher declares
Apache 2.0. The stage downloads its encoder, decoder, joiner and token table,
plus a pinned sherpa-onnx CUDA 12.8 and cuDNN 9 Python wheel. Every file is
SHA-256 checked. The assets total 525,701,236 bytes before extraction. The
Git revision of the installed wheel repository is
`c9f5ed2cf4b2663d23f4e93ab7f642b176dc62e3`.
[Model card](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26)
[CUDA runtime installation](https://k2-fsa.github.io/sherpa/onnx/python/install.html)
[Pinned online recognizer API](https://github.com/k2-fsa/sherpa-onnx/blob/v1.12.23/sherpa-onnx/python/sherpa_onnx/online_recognizer.py)

The recorder uses the archive's resampled 16 kHz PCM, Silero stop and
channel alignment. It feeds packets at their end times on the replay
client's monotonic clock and stamps complete interim replacements after
decode finishes. Batch order uses the immutable pause ID hash. Agent and
caller use separate streaming states. The prior agent state closes after
300 ms of observed quiet and drains with the pinned runtime's documented
EOF padding. Its final decoded text must arrive before the caller's onset.
A missing prior on a two-turn case abstains. A genuine first turn uses
`---`. Caller audio is neither finalized nor synthetically padded.
[Pinned transducer EOF example](https://github.com/k2-fsa/sherpa-onnx/blob/v1.12.23/python-api-examples/online-decode-files.py#L381-L398)

At 300 ms and each subsequent 100 ms decision, the packer considers only
transcript revisions received through that time. Caller resumption and the
800 ms timeout take priority over further application calls. Keep replay
and provider settings separate from the production STT configuration. The
receipt records the fixed input membership, provider settings, trace and
packed digests, and the maximum pacing lag. The first 50 hash-ordered
pauses repeat twice with the same batch composition. The report gives
exact raw and packed text agreement, abstentions, and paired arrival
timing. Fixed co-tenancy isolates run-to-run variation in this harness.

The model and runtime downloads are public. Real HarperValley audio and
interim text stay on Vast. The full-run script copies the private raw trace,
packed decisions, repeat traces and receipt to the project's private R2
prefix and verifies them by SHA-256 readback. Only the aggregate receipt provenance
and coverage belong in notes. This is a provisional ASR condition pending
independent label qualification.
