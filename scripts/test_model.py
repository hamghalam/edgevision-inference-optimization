import torch
from src.model import load_model

model = load_model(pretrained=True)

# Simulate one RGB image: batch=1, channels=3, height=224, width=224
dummy_input = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    output = model(dummy_input)

print("Input shape:", tuple(dummy_input.shape))
print("Output shape:", tuple(output.shape))
print("Predicted class index:", output.argmax(dim=1).item())
print("Inference test: SUCCESS")
