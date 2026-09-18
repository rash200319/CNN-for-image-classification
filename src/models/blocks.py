import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per-channel) followed by a 1x1 pointwise conv.

    A standard KxK conv costs C_in * C_out * K^2 * H * W multiplies.
    Splitting it into a depthwise step (C_in * K^2 * H * W, one filter
    per input channel, no cross-channel mixing) plus a pointwise 1x1
    step (C_in * C_out * H * W, mixes channels) cuts the op count by
    roughly a factor of K^2 * C_out / (K^2 + C_out) - about 8-9x for
    typical 3x3 / 32-256 channel layers (MobileNets).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
