from torchvision import transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm

train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


def calculate_accuracy_and_loss(
    loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: str
) -> tuple[float, float]:
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.inference_mode():
        running_loss = 0
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            running_loss += loss.item()
            preds = logits.argmax(dim=-1).type(torch.long)
            num_correct += (preds == y).sum()
            num_samples += x.shape[0]
    avgloss = running_loss / len(loader)
    acc = (num_correct / num_samples).item()
    model.train()

    return acc, avgloss


def save_best_model(model, accuracy, epoch, best_accuracy, save_path) -> float:
    if accuracy > best_accuracy:
        print(f"New best model found with accuracy: {accuracy:.4f}. Saving model...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
            },
            save_path,
        )
        return accuracy
    return best_accuracy
