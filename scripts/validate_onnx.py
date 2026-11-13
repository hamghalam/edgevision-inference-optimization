import numpy as np
import torch
import onnxruntime as ort

from src.model import load_model


def validate():
    # Load PyTorch model
    model = load_model(pretrained=True)
    model.eval()

    # Generate reproducible input
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 224, 224)

    # PyTorch inference
    with torch.inference_mode():
        pytorch_output = model(dummy_input).numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(
        "models/mobilenet_v2.onnx",
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    onnx_output = session.run(
        None,
        {input_name: dummy_input.numpy()}
    )[0]

    # Numerical comparison
    absolute_error = np.abs(pytorch_output - onnx_output)

    max_error = np.max(absolute_error)
    mean_error = np.mean(absolute_error)

    # Prediction comparison
    pytorch_class = np.argmax(pytorch_output, axis=1)
    onnx_class = np.argmax(onnx_output, axis=1)

    agreement = np.array_equal(pytorch_class, onnx_class)

    print("\n--- PyTorch vs ONNX Validation ---")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape:    {onnx_output.shape}")
    print(f"Maximum absolute error: {max_error:.8f}")
    print(f"Mean absolute error:    {mean_error:.8f}")
    print(f"PyTorch top-1 class: {pytorch_class[0]}")
    print(f"ONNX top-1 class:    {onnx_class[0]}")
    print(f"Top-1 agreement: {agreement}")

    if max_error < 1e-4 and agreement:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION NEEDS INVESTIGATION")


if __name__ == "__main__":
    validate()
