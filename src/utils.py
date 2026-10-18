import torch.nn as nn

from src.data.dataset import NORM_MEAN, NORM_STD
from src.models.pretrained import IMAGENET_MEAN, IMAGENET_STD, PRETRAINED_MODELS, build_pretrained


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, num_classes: int) -> nn.Module:
    from src.models.model_a import ModelA
    from src.models.model_b import ModelB

    if name == "a":
        return ModelA(num_classes=num_classes)
    if name == "b":
        return ModelB(num_classes=num_classes)
    if name in PRETRAINED_MODELS:
        return build_pretrained(name, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def normalization_for(name: str):
    if name in PRETRAINED_MODELS:
        return IMAGENET_MEAN, IMAGENET_STD
    return NORM_MEAN, NORM_STD
