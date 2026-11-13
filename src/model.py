import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def load_model(pretrained: bool = True):
    """
    Load MobileNetV2 with optional ImageNet pretrained weights.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None

    model = mobilenet_v2(weights=weights)
    model.eval()

    return model


if __name__ == "__main__":
    model = load_model()

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("Model: MobileNetV2")
    print(f"Parameters: {total_parameters:,}")
    print(f"Training mode: {model.training}")
