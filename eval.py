import torch.nn as nn
import torch
from model import DenseNet
from config import *
from torchvision.datasets import MNIST, FashionMNIST
from utils import test_transforms, calculate_accuracy_and_loss

if __name__ == "__main__":
    path = "lightning_best_model.pth"

    model = DenseNet(
        num_init_features=NUM_INIT_FEATURES,
        bottleneck_size=BOTTLENECK_SIZE,
        growth_rate=GROWTH_RATE,
    )
    model = model.to(DEVICE)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    test_dataset = FashionMNIST(
        root="~/datasets", train=False, transform=test_transforms, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    acc, loss = calculate_accuracy_and_loss(test_loader, model, DEVICE)

    print(f"Accuracy : {acc} | Loss : {loss}")
