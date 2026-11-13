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

1. Environment and project setup: Completed
2. Model implementation and ONNX export: In progress
3. Qualcomm AI Hub compilation and profiling: Planned
4. Quantization and performance comparison: Planned

## Performance Evaluation

We will evaluate inference latency, model size, numerical accuracy, hardware execution performance, and FP16/INT8 optimization trade-offs.

## License

To be determined.
