import torch
import torch.nn as nn
from layers import DenseBlock,TransitionLayer

    
class DenseNet(nn.Module):
    def __init__(self, num_init_features, bottleneck_size, growth_rate, reduction=0.5, blocks=[6, 12, 24, 16], num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        num_features = num_init_features
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(blocks):
            block = DenseBlock(num_layers, num_features, bottleneck_size, growth_rate)
            self.blocks.append(block)
            num_features += num_layers * growth_rate
            if i != len(blocks) - 1:
                transition = TransitionLayer(num_features, reduction)
                self.transitions.append(transition)
                num_features = int(num_features * reduction)
        self.final_norm = nn.BatchNorm2d(num_features)
        self.pool = nn.AvgPool2d(4)
        self.head = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.pool(self.final_norm(x))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    x = torch.randn(2,1,64,64)
    model = DenseNet(num_init_features=64,bottleneck_size=2,growth_rate=12)
    params = sum([p.numel() for p in model.parameters()])
    print(f"Params: {params}")
    print(model(x).shape)