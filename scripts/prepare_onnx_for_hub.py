import onnx
from pathlib import Path
from onnx.external_data_helper import convert_model_from_external_data

source = Path("models/mobilenet_v2.onnx")
destination = Path("models/mobilenet_v2_embedded.onnx")

print("Loading ONNX model and external weights...")
model = onnx.load(str(source), load_external_data=True)

print("Embedding external weights...")
convert_model_from_external_data(model)

print("Saving self-contained ONNX model...")
onnx.save_model(model, str(destination))

print("Running full ONNX validation...")
onnx.checker.check_model(str(destination), full_check=True)

print("\nPackaging successful!")
print("Output:", destination)
print("File size:", round(destination.stat().st_size / (1024 * 1024), 2), "MB")
print("ONNX validation: PASSED")
