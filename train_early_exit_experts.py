import argparse
import os
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from MobileNetV2 import build_early_exit_mobilenet_v2
from distortion_utils import apply_distortion, distort_half_batch


def split_dataset(dataset, seed: int):
    total = len(dataset)
    n_train = int(0.8 * total)
    n_val = int(0.1 * total)
    n_test = total - n_train - n_val
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


def maybe_distort_batch(images: torch.Tensor, distortion_type: str, mode: str):
    if distortion_type == "pristine":
        return images
    if mode == "train":
        return distort_half_batch(images, distortion_type)
    out = images.clone()
    for i in range(out.size(0)):
        out[i] = apply_distortion(out[i], distortion_type, sigma=None)
    return out


def compute_multi_exit_loss(
    outputs: Dict[str, torch.Tensor], labels: torch.Tensor, criterion: nn.Module
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = {}
    total = 0.0
    for key, logits in outputs.items():
        loss = criterion(logits, labels)
        losses[key] = loss
        total = total + loss
    total = total / len(outputs)
    loss_values = {k: v.item() for k, v in losses.items()}
    return total, loss_values


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return 100.0 * (preds == labels).float().mean().item()


def run_epoch(model, loader, optimizer, criterion, device, expert_type, mode):
    train_mode = mode == "train"
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cloud_acc = 0.0
    batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        images = maybe_distort_batch(images, expert_type, mode)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(images, expert_type=expert_type)
            loss, _ = compute_multi_exit_loss(outputs, labels, criterion)
            if train_mode:
                loss.backward()
                optimizer.step()

        cloud_acc = accuracy_from_logits(outputs["cloud"], labels)
        total_loss += loss.item()
        total_cloud_acc += cloud_acc
        batches += 1

    return total_loss / max(1, batches), total_cloud_acc / max(1, batches)


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = None
        self.wait = 0
        self.stop = False

    def update(self, loss: float):
        if self.best is None or loss < self.best:
            self.best = loss
            self.wait = 0
            return True
        self.wait += 1
        if self.wait >= self.patience:
            self.stop = True
        return False


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_stage(
    model,
    train_loader,
    val_loader,
    expert_type: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device,
    checkpoint_path: str,
    train_backbone: bool,
):
    criterion = nn.CrossEntropyLoss()
    if train_backbone:
        model.unfreeze_backbone()
        params = model.parameters()
    else:
        model.freeze_backbone()
        params = model.parameters_for_expert(expert_type)

    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    stopper = EarlyStopping(patience=patience)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, expert_type=expert_type, mode="train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, expert_type=expert_type, mode="val"
        )
        scheduler.step()
        improved = stopper.update(val_loss)
        print(
            "[%s] epoch %d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%"
            % (expert_type, epoch, train_loss, train_acc, val_loss, val_acc)
        )
        if improved:
            save_checkpoint(checkpoint_path, model, optimizer, epoch)
        if stopper.stop:
            print("[%s] early stopping at epoch %d" % (expert_type, epoch))
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def main():
    parser = argparse.ArgumentParser(description="Train early-exit MobileNetV2 with expert branches.")
    parser.add_argument("--dataset_root", required=True, help="ImageFolder root for pristine training images.")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_pristine", type=int, default=50)
    parser.add_argument("--epochs_expert", type=int, default=30)
    parser.add_argument("--lr_fc", type=float, default=1e-2)
    parser.add_argument("--lr_expert", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root=args.dataset_root, transform=transform)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_early_exit_mobilenet_v2(num_classes=args.num_classes, pretrained_backbone=True).to(device)

    # Stage 1: train pristine expert + backbone.
    pristine_ckpt = os.path.join(args.save_dir, "early_exit_pristine.pt")
    train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        expert_type="pristine",
        epochs=args.epochs_pristine,
        lr=args.lr_fc,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        checkpoint_path=pristine_ckpt,
        train_backbone=True,
    )

    # Stage 2: blur and noise experts, initialized from pristine and trained with frozen backbone.
    for expert_type in ("blur", "noise"):
        model.initialize_expert_from("pristine", expert_type)
        expert_ckpt = os.path.join(args.save_dir, "early_exit_%s.pt" % expert_type)
        train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            expert_type=expert_type,
            epochs=args.epochs_expert,
            lr=args.lr_expert,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
            checkpoint_path=expert_ckpt,
            train_backbone=False,
            )

    # Final checkpoint with all experts.
    final_ckpt = os.path.join(args.save_dir, "early_exit_all_experts.pt")
    save_checkpoint(final_ckpt, model, optimizer=optim.Adam(model.parameters(), lr=1e-3), epoch=0)

    # Quick cloud-head test report with distortion-specific experts.
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for expert_type in ("pristine", "blur", "noise"):
            total_loss = 0.0
            total_acc = 0.0
            n = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                images = maybe_distort_batch(images, expert_type=expert_type, mode="test")
                outputs = model(images, expert_type=expert_type)
                loss = criterion(outputs["cloud"], labels)
                total_loss += loss.item()
                total_acc += accuracy_from_logits(outputs["cloud"], labels)
                n += 1
            print(
                "[test-%s] cloud loss %.4f acc %.2f%%"
                % (expert_type, total_loss / max(1, n), total_acc / max(1, n))
            )


if __name__ == "__main__":
    main()
