import torch.nn as nn

from src.config import MODEL_B_MAX_PARAMS
from src.models.blocks import DepthwiseSeparableConv
from src.utils import count_parameters


class ModelB(nn.Module):
    """Lightweight CNN using depthwise separable convolutions.

    Same 3-stage 64->32->16->8 spatial pipeline as Model A, but each
    conv is depthwise separable and the head uses global average
    pooling instead of a flatten + large FC layer, which is what
    keeps the parameter count small (a flattened 8x8x128 feature map
    into even a 128-unit FC layer alone would cost ~1M params).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32),
            nn.MaxPool2d(2),  # 64 -> 32
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),  # 32 -> 16
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

        n_params = count_parameters(self)
        assert n_params <= MODEL_B_MAX_PARAMS, (
            f"Model B has {n_params} trainable params, exceeds the "
            f"{MODEL_B_MAX_PARAMS} budget"
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
