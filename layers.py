import torch.nn as nn
import torch
from config import DROPOUT


class TransitionLayer(nn.Module):
    def __init__(self, num_features: int, reduction: int = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features,
                int(num_features * reduction),
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DenseLayer(nn.Module):
    def __init__(self, num_features: int, growth_rate: int, bottleneck_size: int):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features,
                bottleneck_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(bottleneck_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Dropout(DROPOUT),
        )

    def forward(self, input: list) -> torch.Tensor:
        prev_features = input
        prev_features = torch.cat(prev_features, dim=1)
        bottleneck_out = self.bottleneck(prev_features)
        out_features = self.conv(bottleneck_out)

        return out_features


class DenseBlock(nn.Module):
    def __init__(
        self, num_layers: int, num_features: int, bottleneck_size: int, growth_rate: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(num_features + i * growth_rate, growth_rate, bottleneck_size)
                for i in range(num_layers)
            ]
        )

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=1)
