import torch
import onnx
from pathlib import Path

from src.model import load_model


def export_model():
    model = load_model(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    output_path = Path("models/mobilenet_v2.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Validate ONNX file structure
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print("ONNX export successful!")
    print("Model:", output_path)
    print("Opset:", 18)
    print("ONNX validation: PASSED")


if __name__ == "__main__":
    export_model()
