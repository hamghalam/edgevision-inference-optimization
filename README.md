# EdgeVision

### Hardware-Aware Deep Learning Optimization and Deployment on Qualcomm Snapdragon Platforms

EdgeVision explores the deployment and optimization of deep learning models on Qualcomm Snapdragon hardware.

## Objectives

- Develop and validate PyTorch computer vision models.
- Export models to ONNX.
- Compile models using Qualcomm AI Hub.
- Run and profile inference on Snapdragon hardware.
- Compare CPU and NPU performance.
- Evaluate FP16 and INT8 quantization.
- Measure latency, model size, and numerical accuracy.

## Initial Target Platform

- Device: Snapdragon X Elite CRD
- Chipset: Snapdragon X Elite (SC8380XP)
- Platform: Qualcomm AI Hub

## Initial Model

MobileNetV2 pretrained on ImageNet.

## Project Structure

- `src/`: Model implementation and inference
- `models/`: Exported model artifacts
- `scripts/`: Export and benchmarking scripts
- `benchmarks/`: Performance and accuracy results
- `docs/`: Technical documentation

## Development Environment

- Python 3.12
- PyTorch
- Torchvision
- ONNX
- ONNX Runtime
- Qualcomm AI Hub SDK

## Project Status

### Completed
1. Environment and project setup.
2. MobileNetV2 PyTorch implementation.
3. ONNX export and model validation.
4. PyTorch vs. original ONNX numerical validation.
5. Qualcomm AI Hub compilation for Snapdragon X Elite CRD.
6. Remote inference on Snapdragon X Elite CRD.
7. Hardware profiling confirming NPU execution for all 104 profiled operations.

### In Progress
1. Numerical accuracy comparison between PyTorch and Snapdragon inference.
2. Repeated inference benchmarking and performance analysis.

### Planned
1. FP16 and INT8 quantization.
2. Accuracy, latency, and model-size comparison across optimization configurations.
3. Further investigation of hardware-aware deployment and optimization.

## Performance Evaluation

Initial Qualcomm AI Hub profiling results:

| Metric | Result |
|---|---:|
| Target device | Snapdragon X Elite CRD |
| Profiled operations assigned to NPU | 104 / 104 |
| Qualcomm estimated inference time | 0.541 ms |
| Estimated inference peak memory | 36.81 MB |
| First-load time | 903.627 ms |
| Warm-load time | 519.271 ms |

The inference time is a Qualcomm profiling estimate, not independently measured end-to-end application latency. The initial benchmark uses a single synthetic input.

See [the initial benchmark report](docs/benchmark-report.md) for details.

## License

To be determined.
