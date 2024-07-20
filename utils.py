from torchvision import transforms
import torch
import torch.nn.functional as F

train_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

def calculate_accuracy_and_loss(loader: torch.utils.data.DataLoader,model: torch.nn.Module, device: str):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.inference_mode():
        running_loss = 0
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits,y)
            running_loss += loss.item()
            preds = logits.argmax(dim=-1).type(torch.long)
            num_correct += (preds == y).sum()
            num_samples += x.shape[0]
    avgloss = running_loss / len(loader)
    acc = (num_correct / num_samples).item()     
    model.train()

    return acc,avgloss

