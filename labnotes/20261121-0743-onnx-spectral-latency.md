# ONNX spectral frontend latency

- The [int8 window probe](20261121-0736-ami-additional-turns.md)
  attributed the majority of latency to the ONNX STFT operator.
  Training and checkpoint files can stay compatible if the change is
  confined to the export graph.
- Prototyped replacing the STFT with gathered window frames and a
  native batched DFT. Synthetic batch-one p99 increased from about
  2.57 ms to 3.72 ms at 3 s. Two matrix-factorized Fourier graphs were
  also slower, with p99 exceeding 5.5 ms. Kept the fixed convolution
  approach for further validation. See the
  [ONNX DFT](https://onnx.ai/onnx/operators/onnx__DFT.html) and
  [STFT](https://onnx.ai/onnx/operators/onnx__STFT.html) operator contracts.
- Added `superturn export CHECKPOINT --spectrum convolution` as an
  explicit option. The fixed, non-trained filter bank applies the same
  400-point Hann window and projects to the first 129 bins of the
  512-point DFT. The hop is 160 samples. Keep the downstream processing
  and learned checkpoint fixed. The optional exporter checks float
  scores against the original frontend on generated noise, silence,
  sparse audio and tones.
- Local tests passed 51 with 14 skips. Both generated GPU-machine
  architecture fixture exports passed using no STFT and under 1 MB.
  The standard export path still uses STFT.
- Launched a validation-only comparison on 89 real G.711 clips from
  fold 0's calibration set, using the same three fold-0 GRU checkpoints
  and fixed thresholds as the earlier CPU check. Benchmark each int8
  graph with 1,000 batch-one calls on one pinned CPU core.

## Results

| Window | STFT p99 | Convolution p99 | Convolution int8 bytes |
|---|---:|---:|---:|
| 1 s | 0.895 ms | 0.478 ms | 713,118 |
| 3 s | 2.559 ms | 1.201 ms | 713,122 |
| 6 s | 5.062 ms | 2.306 ms | 713,122 |

- On 89 G.711 clips, maximum int8 probability delta was 5.96e-8.
  Each model kept the same int8 AUC and decisions at its previous
  fixed threshold. The Fourier bank is a float32 constant of shape
  258 × 1 × 400. The learned weights retain their int8 packing.
  The 3 s export is below the 2 ms target on this machine. The
  6 s export still exceeds the target.
- Rendered the same 89 reference clips as wideband, telephone plus
  noise at 10 dB SNR, and quiet telephone (0.03 gain). Across all
  three windows and those conditions, maximum float probability
  delta was 5.96e-8. Maximum int8 delta was 8.69e-5, on a wideband
  3 s example. These checks assess export numerics on validation,
  with no implication for recognition accuracy at the lower level.
- Reports `reports/ami-convolution-spectrum.json` and
  `reports/ami-convolution-spectrum-conditions.json`. R2 verification
  matched 527 exports and 47 reports with zero differences. This was
  an export and validation-only comparison. A candidate still needs
  a frozen-test benchmark and the quality target.
- The final server suite passed 65 tests in 41.64 s. Astra reviewed
  spectral equivalence, batched export, fixed coefficients, artifact
  size enforcement, and compatibility with the standard path.
