import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from distortionNet import DistortionNet
from distortion_utils import apply_distortion


LABEL_TO_NAME = {0: "pristine", 1: "blur", 2: "noise"}


class DistortionClassificationDataset(Dataset):
    """
    Builds a synthetic 3-class dataset from pristine images.
    Each sample is randomly assigned to pristine/blur/noise.
    """

    def __init__(self, base_dataset: Dataset, split: str = "train"):
        self.base_dataset = base_dataset
        self.split = split

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, _ = self.base_dataset[index]
        label = random.randint(0, 2)
        distortion_name = LABEL_TO_NAME[label]
        if self.split == "train":
            image = apply_distortion(image, distortion_name, sigma=None)
        elif self.split in ("val", "test"):
            # Keep deterministic behavior by fixed representative levels.
            sigma = None
            if distortion_name == "blur":
                sigma = 3
            if distortion_name == "noise":
                sigma = 20
            image = apply_distortion(image, distortion_name, sigma=sigma)
        return image, label


class EarlyStopping:
    def __init__(self, patience: int, save_path: str):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = None
        self.wait = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                },
                self.save_path,
            )
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.stop = True


def run_epoch(model, loader, optimizer, criterion, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train Fourier-spectrum distortion classifier.")
    parser.add_argument("--root_path", required=True, help="Path to pristine dataset root (ImageFolder).")
    parser.add_argument("--save_path", default="checkpoints/distortion_classifier.pt")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )
    full_dataset = datasets.ImageFolder(root=args.root_path, transform=transform)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_base, val_base, test_base = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataset = DistortionClassificationDataset(train_base, split="train")
    val_dataset = DistortionClassificationDataset(val_base, split="val")
    test_dataset = DistortionClassificationDataset(test_base, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistortionNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    stopper = EarlyStopping(patience=args.patience, save_path=args.save_path)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, train_mode=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, device, train_mode=False)
        scheduler.step(val_loss)
        stopper(val_loss, model, optimizer, epoch)
        print(
            "Epoch %d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%"
            % (epoch, train_loss, train_acc, val_loss, val_acc)
        )
        if stopper.stop:
            print("Early stopping at epoch %d" % epoch)
            break

    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, train_mode=False)
    print("Test loss %.4f | Test acc %.2f%%" % (test_loss, test_acc))


if __name__ == "__main__":
    main()
