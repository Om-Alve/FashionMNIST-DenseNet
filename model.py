import torch
import torch.nn as nn
from config import DROPOUT

class TransitionLayer(nn.Module):
    def __init__(self,num_features,reduction):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features,int(num_features * reduction),kernel_size=1,stride=1,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )
    def forward(self,x):
        return self.conv(x)
    
class DenseLayer(nn.Module):
    def __init__(self,num_features,growth_rate,bottleneck_size,dropout=0.2):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features,bottleneck_size * growth_rate,kernel_size=1,stride=1,bias=False),
        ) 
        self.conv = nn.Sequential(
            nn.BatchNorm2d(bottleneck_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_size * growth_rate, growth_rate,kernel_size=3,stride=1,padding=1,bias=False),
            nn.Dropout(dropout),
        )
        
    def forward(self,input):
        prev_features = input
        prev_features = torch.cat(prev_features,dim=1)
        bottleneck_out = self.bottleneck(prev_features)
        out_features = self.conv(bottleneck_out)
        
        return out_features

class DenseBlock(nn.Module):
    def __init__(self,num_layers,num_features,bottleneck_size,growth_rate):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(num_features + i * growth_rate,growth_rate,bottleneck_size) for i in range(num_layers)])
    
    def forward(self,init_features):
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features,dim=1)
    
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