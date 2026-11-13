# EdgeVision – Snapdragon X Elite Benchmark Report

**Model:** MobileNetV2
**Target device:** Qualcomm Snapdragon X Elite CRD
**Execution platform:** Qualcomm AI Hub
**Status:** Initial benchmark completed

## 1. Objective

Evaluate the deployment of a PyTorch MobileNetV2 model on Qualcomm Snapdragon hardware using ONNX export, Qualcomm AI Hub compilation, and hardware profiling.

## 2. Model and validation

| Item | Result |
|---|---:|
| Architecture | MobileNetV2 |
| Parameters | 3,504,872 |
| Input shape | `(1, 3, 224, 224)` |
| Output shape | `(1, 1000)` |
| ONNX opset | 18 |
| PyTorch vs. original ONNX maximum absolute error | 0.00000453 |
| Top-1 agreement | Yes |

The original ONNX model passed validation against the PyTorch implementation.

## 3. Qualcomm deployment

- ONNX model successfully compiled for Snapdragon X Elite CRD.
- Remote inference job completed successfully.
- Qualcomm profiling returned 104 execution entries.
- All 104 entries were assigned to the NPU.
- No CPU or GPU operations were reported in the execution profile.

## 4. Performance results

| Metric | Measured/reported value |
|---|---:|
| Estimated inference time | 541 µs (0.541 ms) |
| Estimated inference peak memory | 36.81 MB |
| First-load time | 903.627 ms |
| Warm-load time | 519.271 ms |
| First-load peak memory | 271.66 MB |
| Warm-load peak memory | 36.76 MB |

The inference time is Qualcomm's profile estimate. It should not be interpreted as independently measured end-to-end application latency.

## 5. Profiling observations

The profile contained 104 operations, all assigned to the Qualcomm NPU. The largest reported operation times included:

- `n4`: 369 µs
- `n4`: 256 µs
- `Transpose`: 190 µs
- `n4`: 124 µs

These are compiled-operation names rather than necessarily direct mappings to individual PyTorch layers.

## 6. Reproducibility

- Qualcomm compile job: `jp1merrnp`
- Qualcomm target model: `mmr63ep2r`
- Qualcomm inference job: `jp42vqdlg`
- Input dataset: `d78zg6gy7`
- Profile artifact: `benchmarks/snapdragon_profile.json`

## 7. Limitations and next steps

This is an initial single-input benchmark using a synthetic random input tensor. The following work remains:

1. Compare Snapdragon inference outputs numerically against PyTorch.
2. Benchmark multiple inputs and repeated runs.
3. Investigate FP16 and INT8 quantization.
4. Compare model size, accuracy, latency, and memory.
5. Evaluate end-to-end deployment performance.

**Conclusion:** The initial experiment successfully compiled MobileNetV2 for Snapdragon X Elite and demonstrated NPU-only placement across all 104 profiled operations. Further accuracy and repeated-performance testing is required before drawing broader conclusions about deployment performance.
