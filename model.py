import torch
import torch.nn as nn
from config import DROPOUT

class Classifier(nn.Module):
    def __init__(self,inchannels :int = 1,num_classes :int = 10):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=32,kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(DROPOUT),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(DROPOUT),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(DROPOUT),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(256,128),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(128,num_classes),
        )
    
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        return self.convnet(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    x = torch.randn(1,1,64,64)
    model = Classifier()
    params = sum([p.numel() for p in model.parameters()])
    print(f"Params: {params}")
    print(model(x).shape)