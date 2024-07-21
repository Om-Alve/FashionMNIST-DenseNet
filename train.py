import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from utils import (
    train_transforms,
    calculate_accuracy_and_loss,
    test_transforms,
    save_best_model,
)
from config import *
from model import DenseNet
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    train_dataset = FashionMNIST(
        root="~/datasets", train=True, transform=train_transforms, download=True
    )
    test_dataset = FashionMNIST(
        root="~/datasets", train=False, transform=test_transforms, download=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    model = DenseNet(num_init_features=64, bottleneck_size=2, growth_rate=12).to(DEVICE)
    opt = torch.optim.AdamW(lr=LEARNING_RATE, params=model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-2,total_steps=NUM_EPOCHS * len(train_dataloader),anneal_strategy='cos',max_momentum=0.90)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=NUM_EPOCHS * len(train_dataloader), eta_min=1e-4
    )
    writer = SummaryWriter("runs/fashion_MNIST_DENSENET")
    best_accuracy = 0
    save_path = "best_model_.pth"
    for i in range(NUM_EPOCHS):
        loader = tqdm(train_dataloader, leave=False)
        loader.set_description(f"Epoch: {i}")
        running_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
            running_loss += loss.item()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        test_acc, test_loss = calculate_accuracy_and_loss(
            test_dataloader, model, DEVICE
        )
        writer.add_scalar("training_loss", running_loss / len(train_dataloader), i)
        writer.add_scalar("testing_loss", test_loss, i)
        writer.add_scalar("test accuracy", test_acc, i)
        best_accuracy = save_best_model(model, test_acc, i, best_accuracy, save_path)
