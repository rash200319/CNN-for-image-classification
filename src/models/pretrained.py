import torch.nn as nn
from torchvision import models

PRETRAINED_MODELS = ("mobilenet_v2", "efficientnet_b0")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_pretrained(name: str, num_classes: int) -> nn.Module:
    """ImageNet-pretrained backbone with the final classifier replaced for `num_classes`."""
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown pretrained model: {name}")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
