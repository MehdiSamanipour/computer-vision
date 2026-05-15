# ==========================================================
# Project 15: Early-Exit Deep Neural Networks
# Improvement focus:
#   Grid search over early-exit branch placement.
#
# Dataset:
#   Caltech-256 / 257 classes
#
# Main commands:
#   checkdata
#   train
#   gridsearch
#   test
#   infer
# ==========================================================


# ==========================================================
# PART 1) Import packages
# What this part does:
#   Loads all required Python, PyTorch, torchvision,
#   plotting, data handling, and utility packages.
# ==========================================================

import argparse
import json
import os
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ==========================================================
# PART 2) Reproducibility and helper functions
# What this part does:
#   Fixes random seeds, creates output folders, parses
#   early-exit placement strings, and synchronizes GPU timing.
# ==========================================================

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(force_cpu: bool = False) -> torch.device:
    if torch.cuda.is_available() and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_output_dirs(output_dir: str) -> Dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "models": root / "models",
        "plots": root / "plots",
        "csv": root / "csv",
        "json": root / "json",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def parse_exit_indices(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("exit_indices must contain exactly 3 integers, example: 3,6,13")
    if len(set(parts)) != 3:
        raise ValueError("exit_indices must be unique.")
    return tuple(sorted(parts))


def parse_placements(text: str) -> List[Tuple[int, int, int]]:
    if text.strip().lower() == "default":
        # Candidate placements in MobileNetV2 feature blocks.
        # Lower numbers = earlier exits, higher numbers = deeper exits.
        return [
            (2, 5, 10),
            (3, 6, 13),
            (4, 8, 14),
            (5, 9, 15),
            (6, 10, 16),
            (7, 12, 17),
        ]

    candidates = []
    for candidate in text.split(";"):
        candidate = candidate.strip()
        if candidate:
            candidates.append(parse_exit_indices(candidate))
    if not candidates:
        raise ValueError("No valid placements found.")
    return candidates


def placement_tag(exit_indices: Sequence[int]) -> str:
    return "place_" + "_".join(str(x) for x in exit_indices)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ==========================================================
# PART 3) Dataset split and optional distortion transforms
# What this part does:
#   Loads Caltech-256 from ImageFolder, creates class-balanced
#   train/validation/test splits, and optionally applies
#   Gaussian blur or Gaussian noise during evaluation.
# ==========================================================

class AddGaussianNoise(object):
    def __init__(self, sigma_255: float):
        self.std = float(sigma_255) / 255.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_transforms(
    image_size: int,
    eval_distortion: str = "none",
    blur_sigma: float = 2.0,
    noise_sigma: float = 10.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_steps = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
    ]

    if eval_distortion == "blur":
        kernel_size = int(4 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        eval_steps.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=(blur_sigma, blur_sigma)))

    eval_steps.append(transforms.ToTensor())

    if eval_distortion == "noise":
        eval_steps.append(AddGaussianNoise(noise_sigma))

    eval_steps.append(transforms.Normalize(imagenet_mean, imagenet_std))

    eval_transform = transforms.Compose(eval_steps)

    return train_transform, eval_transform


def stratified_split_indices(
    targets: Sequence[int],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    class_to_indices = defaultdict(list)

    for idx, y in enumerate(targets):
        class_to_indices[int(y)].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for _, idxs in class_to_indices.items():
        rng.shuffle(idxs)
        n = len(idxs)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def reduce_indices(indices: List[int], fraction: float, seed: int) -> List[int]:
    if fraction >= 1.0:
        return indices
    rng = random.Random(seed)
    idxs = list(indices)
    rng.shuffle(idxs)
    n = max(1, int(len(idxs) * fraction))
    return idxs[:n]


def make_dataloaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    subset_fraction: float = 1.0,
    eval_distortion: str = "none",
    blur_sigma: float = 2.0,
    noise_sigma: float = 10.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        raise FileNotFoundError("Dataset root not found: {}".format(data_root_path.resolve()))

    train_tf, eval_tf = get_transforms(
        image_size=image_size,
        eval_distortion=eval_distortion,
        blur_sigma=blur_sigma,
        noise_sigma=noise_sigma,
    )

    base = datasets.ImageFolder(str(data_root_path))
    train_idx, val_idx, test_idx = stratified_split_indices(
        base.targets,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_idx = reduce_indices(train_idx, subset_fraction, seed)
    val_idx = reduce_indices(val_idx, subset_fraction, seed + 1)
    test_idx = reduce_indices(test_idx, subset_fraction, seed + 2)

    train_all = datasets.ImageFolder(str(data_root_path), transform=train_tf)
    eval_all = datasets.ImageFolder(str(data_root_path), transform=eval_tf)

    train_ds = Subset(train_all, train_idx)
    val_ds = Subset(eval_all, val_idx)
    test_ds = Subset(eval_all, test_idx)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader, base.classes


# ==========================================================
# PART 4) Early-exit head
# What this part does:
#   Defines a small classifier attached to an intermediate
#   feature map. Each branch predicts the full class label.
# ==========================================================

class ExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.20):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ==========================================================
# PART 5) MobileNetV2 with configurable early exits
# What this part does:
#   Builds a MobileNetV2 backbone and attaches three early
#   exits at user-selected feature-block indices.
# ==========================================================

class EarlyExitMobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        exit_indices: Tuple[int, int, int] = (3, 6, 13),
        pretrained: bool = True,
        dropout: float = 0.20,
        image_size: int = 224,
    ):
        super().__init__()

        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            base = models.mobilenet_v2(weights=weights)
        else:
            base = models.mobilenet_v2(weights=None)

        self.features = base.features
        self.exit_indices = tuple(sorted(exit_indices))
        self.num_classes = num_classes
        self.image_size = image_size

        max_index = len(self.features) - 1
        for idx in self.exit_indices:
            if idx < 0 or idx > max_index:
                raise ValueError("Exit index {} outside MobileNetV2 feature range 0..{}".format(idx, max_index))

        exit_channels = self._infer_exit_channels(image_size=image_size)

        self.exit_heads = nn.ModuleList([
            ExitHead(ch, num_classes, dropout=dropout) for ch in exit_channels
        ])

        self.final_exit = ExitHead(1280, num_classes, dropout=dropout)

    def _infer_exit_channels(self, image_size: int) -> List[int]:
        was_training = self.training
        self.eval()

        channels = []
        x = torch.zeros(1, 3, image_size, image_size)

        with torch.no_grad():
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if idx in self.exit_indices:
                    channels.append(int(x.shape[1]))

        if was_training:
            self.train()

        if len(channels) != len(self.exit_indices):
            raise RuntimeError("Could not infer all exit channels.")

        return channels

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        logits = []
        head_idx = 0

        for idx, layer in enumerate(self.features):
            x = layer(x)

            if idx in self.exit_indices:
                logits.append(self.exit_heads[head_idx](x))
                head_idx += 1

        logits.append(self.final_exit(x))
        return logits

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        return self.final_exit(x)

    @torch.no_grad()
    def predict_one_early_exit(
        self,
        x: torch.Tensor,
        thresholds: Sequence[float],
    ) -> Tuple[int, int, float]:
        head_idx = 0

        for idx, layer in enumerate(self.features):
            x = layer(x)

            if idx in self.exit_indices:
                logits = self.exit_heads[head_idx](x)
                prob = torch.softmax(logits, dim=1)
                conf, pred = prob.max(dim=1)

                if conf.item() >= thresholds[head_idx]:
                    return int(pred.item()), head_idx + 1, float(conf.item())

                head_idx += 1

        logits = self.final_exit(x)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        return int(pred.item()), len(self.exit_indices) + 1, float(conf.item())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_all(x)


# ==========================================================
# PART 6) Early-exit policy and metrics
# What this part does:
#   Applies confidence thresholds to decide whether a sample
#   exits early or continues to the next branch.
# ==========================================================

def policy_predict_batch(
    logits_list: List[torch.Tensor],
    thresholds: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = [torch.softmax(l, dim=1) for l in logits_list]
    batch_size = probs[0].shape[0]
    device = probs[0].device

    final_exit_index = len(probs) - 1

    pred = torch.zeros(batch_size, dtype=torch.long, device=device)
    conf_out = torch.zeros(batch_size, dtype=torch.float32, device=device)
    chosen_exit = torch.full((batch_size,), final_exit_index, dtype=torch.long, device=device)
    undecided = torch.ones(batch_size, dtype=torch.bool, device=device)

    for i in range(final_exit_index):
        conf, cls = probs[i].max(dim=1)
        take = undecided & (conf >= thresholds[i])

        pred[take] = cls[take]
        conf_out[take] = conf[take]
        chosen_exit[take] = i
        undecided[take] = False

    final_conf, final_cls = probs[final_exit_index].max(dim=1)
    pred[undecided] = final_cls[undecided]
    conf_out[undecided] = final_conf[undecided]

    return pred, chosen_exit, conf_out


def cost_points_from_exit_indices(exit_indices: Sequence[int], num_feature_blocks: int = 19) -> List[float]:
    costs = []
    for idx in exit_indices:
        costs.append(float(idx + 1) / float(num_feature_blocks))
    costs.append(1.0)
    return costs


@torch.no_grad()
def evaluate_model(
    model: EarlyExitMobileNetV2,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    criterion: Optional[nn.Module] = None,
    loss_weights: Optional[Sequence[float]] = None,
    desc: str = "Evaluating",
) -> Dict:
    model.eval()

    total = 0
    policy_correct = 0
    exit_correct = [0, 0, 0, 0]
    exit_counts = [0, 0, 0, 0]
    running_loss = 0.0

    if loss_weights is None:
        loss_weights = [0.25, 0.50, 0.75, 1.00]

    costs = cost_points_from_exit_indices(model.exit_indices, num_feature_blocks=len(model.features))

    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model.forward_all(x)

        if criterion is not None:
            loss = 0.0
            for i, logit in enumerate(logits):
                loss = loss + float(loss_weights[i]) * criterion(logit, y)
            loss = loss / sum(loss_weights)
            running_loss += float(loss.item()) * y.size(0)

        for i in range(4):
            pred_i = logits[i].argmax(dim=1)
            exit_correct[i] += int((pred_i == y).sum().item())

        policy_pred, chosen_exit, _ = policy_predict_batch(logits, thresholds)
        policy_correct += int((policy_pred == y).sum().item())

        for i in range(4):
            exit_counts[i] += int((chosen_exit == i).sum().item())

        total += int(y.size(0))

    exit_acc = [c / max(total, 1) for c in exit_correct]
    policy_acc = policy_correct / max(total, 1)
    avg_exit = sum((i + 1) * c for i, c in enumerate(exit_counts)) / max(total, 1)
    avg_cost = sum(costs[i] * exit_counts[i] for i in range(4)) / max(total, 1)
    avg_loss = running_loss / max(total, 1) if criterion is not None else None

    return {
        "num_samples": total,
        "loss": avg_loss,
        "policy_acc": policy_acc,
        "exit_acc": exit_acc,
        "exit_counts": exit_counts,
        "avg_exit": avg_exit,
        "avg_depth_cost_ratio": avg_cost,
        "cost_points": costs,
    }


# ==========================================================
# PART 7) Training functions
# What this part does:
#   Trains all exits jointly. Each exit has its own loss.
#   The total loss is a weighted sum of all exit losses.
# ==========================================================

def build_optimizer(
    model: EarlyExitMobileNetV2,
    lr_backbone: float,
    lr_heads: float,
    weight_decay: float,
    freeze_backbone: bool = False,
) -> torch.optim.Optimizer:
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = []

    for h in model.exit_heads:
        head_params.extend(list(h.parameters()))
    head_params.extend(list(model.final_exit.parameters()))

    params = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_heads},
    ]

    return torch.optim.AdamW(params, weight_decay=weight_decay)


def train_one_epoch(
    model: EarlyExitMobileNetV2,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_weights: Sequence[float],
    use_amp: bool,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    n_seen = 0

    pbar = tqdm(loader, desc="Epoch {}/{}".format(epoch, epochs))

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model.forward_all(x)

            loss = 0.0
            for i, logit in enumerate(logits):
                loss = loss + float(loss_weights[i]) * criterion(logit, y)
            loss = loss / sum(loss_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * y.size(0)
        n_seen += int(y.size(0))

        pbar.set_postfix(loss="{:.4f}".format(float(loss.item())))

    return running_loss / max(n_seen, 1)


def plot_training_curves(history: pd.DataFrame, path: Path, title_suffix: str = "") -> None:
    if history.empty:
        return

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    if "val_loss" in history.columns:
        plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves {}".format(title_suffix))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_policy_acc"] * 100.0, label="Policy Acc")
    plt.plot(history["epoch"], history["val_exit4_acc"] * 100.0, label="Final Exit Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy {}".format(title_suffix))
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_model_for_placement(
    args,
    exit_indices: Tuple[int, int, int],
    tag: str,
    save_as_main_best: bool = False,
) -> Dict:
    seed_everything(args.seed, deterministic=args.deterministic)

    paths = make_output_dirs(args.output_dir)
    device = get_device(args.cpu)

    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Using data_root:", str(Path(args.data_root).resolve()))
    print("Using exit_indices:", exit_indices)

    train_loader, val_loader, _, classes = make_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        subset_fraction=args.subset_fraction,
        eval_distortion="none",
    )

    model = EarlyExitMobileNetV2(
        num_classes=len(classes),
        exit_indices=exit_indices,
        pretrained=args.pretrained,
        dropout=args.dropout,
        image_size=args.image_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = build_optimizer(
        model=model,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    thresholds = [args.t1, args.t2, args.t3]
    loss_weights = [args.w1, args.w2, args.w3, args.w4]
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_policy_acc = -1.0
    best_cost = 999.0
    best_epoch = -1
    best_path = paths["models"] / "{}_best.pth".format(tag)
    last_path = paths["models"] / "{}_last.pth".format(tag)

    history_rows = []
    patience_counter = 0
    best_val_loss = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            loss_weights=loss_weights,
            use_amp=args.amp,
            epoch=epoch,
            epochs=args.epochs,
        )

        scheduler.step()

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=thresholds,
            criterion=criterion,
            loss_weights=loss_weights,
            desc="Validation",
        )

        row = {
            "tag": tag,
            "epoch": epoch,
            "exit_indices": ",".join(str(x) for x in exit_indices),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_policy_acc": val_metrics["policy_acc"],
            "val_exit1_acc": val_metrics["exit_acc"][0],
            "val_exit2_acc": val_metrics["exit_acc"][1],
            "val_exit3_acc": val_metrics["exit_acc"][2],
            "val_exit4_acc": val_metrics["exit_acc"][3],
            "val_avg_exit": val_metrics["avg_exit"],
            "val_avg_depth_cost_ratio": val_metrics["avg_depth_cost_ratio"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_heads": optimizer.param_groups[1]["lr"],
        }
        history_rows.append(row)

        print(
            "Epoch {:02d} | loss={:.4f} | val_acc={:.4f} | final_acc={:.4f} | avg_exit={:.3f} | cost={:.3f}".format(
                epoch,
                train_loss,
                val_metrics["policy_acc"],
                val_metrics["exit_acc"][3],
                val_metrics["avg_exit"],
                val_metrics["avg_depth_cost_ratio"],
            )
        )

        improved = False
        if val_metrics["policy_acc"] > best_policy_acc:
            improved = True
        elif abs(val_metrics["policy_acc"] - best_policy_acc) < 1e-9 and val_metrics["avg_depth_cost_ratio"] < best_cost:
            improved = True

        if improved:
            best_policy_acc = val_metrics["policy_acc"]
            best_cost = val_metrics["avg_depth_cost_ratio"]
            best_epoch = epoch

            checkpoint = {
                "model_state": model.state_dict(),
                "classes": classes,
                "exit_indices": list(exit_indices),
                "thresholds": thresholds,
                "image_size": args.image_size,
                "dropout": args.dropout,
                "num_classes": len(classes),
                "best_epoch": best_epoch,
                "best_val_policy_acc": best_policy_acc,
                "best_val_avg_depth_cost_ratio": best_cost,
                "args": vars(args),
            }

            torch.save(checkpoint, best_path)
            print("Saved best checkpoint:", best_path)

            if save_as_main_best:
                shutil.copy2(best_path, paths["models"] / "best_model.pth")

        current_val_loss = val_metrics["loss"]
        if current_val_loss is not None:
            if best_val_loss is None or current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print("Early stopping: validation loss did not improve for {} epochs.".format(args.early_stop_patience))
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "exit_indices": list(exit_indices),
            "thresholds": thresholds,
            "image_size": args.image_size,
            "dropout": args.dropout,
            "num_classes": len(classes),
            "last_epoch": history_rows[-1]["epoch"] if history_rows else 0,
            "args": vars(args),
        },
        last_path,
    )

    history = pd.DataFrame(history_rows)
    history_csv = paths["csv"] / "{}_history.csv".format(tag)
    history.to_csv(history_csv, index=False)

    plot_path = paths["plots"] / "{}_training_curves.png".format(tag)
    plot_training_curves(history, plot_path, title_suffix="({})".format(tag))

    summary = {
        "tag": tag,
        "exit_indices": list(exit_indices),
        "best_epoch": best_epoch,
        "best_val_policy_acc": best_policy_acc,
        "best_val_avg_depth_cost_ratio": best_cost,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(history_csv),
        "training_plot": str(plot_path),
    }

    save_json(summary, paths["json"] / "{}_train_summary.json".format(tag))

    print("Training finished for:", tag)
    print("Best val policy accuracy:", best_policy_acc)
    print("Best checkpoint:", best_path)

    return summary


# ==========================================================
# PART 8) Grid search over branch placement
# What this part does:
#   Trains several early-exit placement candidates and chooses
#   the best one by validation accuracy first, then by lower
#   computational cost ratio.
# ==========================================================

def run_gridsearch(args) -> None:
    paths = make_output_dirs(args.output_dir)
    placements = parse_placements(args.placements)

    summaries = []

    print("Grid search candidates:")
    for p in placements:
        print("  ", p)

    for exit_indices in placements:
        tag = placement_tag(exit_indices)

        summary = train_model_for_placement(
            args=args,
            exit_indices=exit_indices,
            tag=tag,
            save_as_main_best=False,
        )

        summaries.append(summary)

    df = pd.DataFrame(summaries)

    df_sorted = df.sort_values(
        by=["best_val_policy_acc", "best_val_avg_depth_cost_ratio"],
        ascending=[False, True],
    ).reset_index(drop=True)

    results_csv = paths["csv"] / "gridsearch_results.csv"
    df_sorted.to_csv(results_csv, index=False)

    best = df_sorted.iloc[0].to_dict()
    best_checkpoint = Path(best["best_checkpoint"])
    main_best = paths["models"] / "best_model.pth"

    shutil.copy2(best_checkpoint, main_best)

    save_json(
        {
            "best_candidate": best,
            "all_candidates": summaries,
            "selection_rule": "highest validation policy accuracy, tie-break by lowest average depth cost ratio",
            "main_best_checkpoint": str(main_best),
        },
        paths["json"] / "gridsearch_summary.json",
    )

    plot_gridsearch_results(df_sorted, paths["plots"] / "gridsearch_accuracy_cost.png")

    print("Grid search finished.")
    print("Saved:", results_csv)
    print("Best placement:", best["exit_indices"])
    print("Main best checkpoint:", main_best)


def plot_gridsearch_results(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return

    labels = [str(x) for x in df["exit_indices"]]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.plot(x, df["best_val_policy_acc"] * 100.0, marker="o", label="Best Val Policy Acc (%)")
    plt.plot(x, df["best_val_avg_depth_cost_ratio"] * 100.0, marker="s", label="Avg Cost Ratio (%)")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Exit placement")
    plt.ylabel("Percent")
    plt.title("Grid Search: Accuracy vs Cost Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ==========================================================
# PART 9) Latency benchmark and threshold sweep
# What this part does:
#   Tests different threshold settings and measures:
#   accuracy, exit distribution, average cost, and latency.
# ==========================================================

@torch.no_grad()
def benchmark_latency(
    model: EarlyExitMobileNetV2,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    max_samples: int = 300,
) -> Dict:
    model.eval()

    full_times = []
    early_times = []

    seen = 0

    for x, _ in loader:
        for i in range(x.shape[0]):
            if seen >= max_samples:
                break

            xi = x[i:i + 1].to(device, non_blocking=True)

            device_sync(device)
            t0 = time.perf_counter()
            _ = model.forward_final(xi)
            device_sync(device)
            t1 = time.perf_counter()
            full_times.append((t1 - t0) * 1000.0)

            device_sync(device)
            t2 = time.perf_counter()
            _ = model.predict_one_early_exit(xi, thresholds)
            device_sync(device)
            t3 = time.perf_counter()
            early_times.append((t3 - t2) * 1000.0)

            seen += 1

        if seen >= max_samples:
            break

    full_ms = float(np.mean(full_times)) if full_times else None
    early_ms = float(np.mean(early_times)) if early_times else None

    if full_ms is not None and early_ms is not None and early_ms > 0:
        speedup = full_ms / early_ms
    else:
        speedup = None

    return {
        "latency_samples": seen,
        "lat_full_ms": full_ms,
        "lat_early_ms": early_ms,
        "speedup_x": speedup,
    }


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    force_dropout: Optional[float] = None,
) -> Tuple[EarlyExitMobileNetV2, List[str], Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    classes = ckpt["classes"]
    exit_indices = tuple(int(x) for x in ckpt["exit_indices"])
    image_size = int(ckpt.get("image_size", 224))
    dropout = float(ckpt.get("dropout", 0.20))

    if force_dropout is not None:
        dropout = force_dropout

    model = EarlyExitMobileNetV2(
        num_classes=len(classes),
        exit_indices=exit_indices,
        pretrained=False,
        dropout=dropout,
        image_size=image_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, classes, ckpt


def threshold_sets_from_args(args, checkpoint_thresholds: Sequence[float]) -> List[List[float]]:
    sets = [
        [0.99, 0.97, 0.95],
        list(checkpoint_thresholds),
        [0.90, 0.85, 0.80],
        [0.80, 0.75, 0.70],
        [0.70, 0.60, 0.50],
    ]

    if args.custom_thresholds:
        sets.append([float(x.strip()) for x in args.custom_thresholds.split(",")])

    unique = []
    seen = set()

    for s in sets:
        key = tuple(round(x, 4) for x in s)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def run_test(args) -> None:
    paths = make_output_dirs(args.output_dir)

    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = str(paths["models"] / "best_model.pth")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    device = get_device(args.cpu)

    print("Using device:", device)
    print("Checkpoint:", checkpoint_path)

    model, classes, ckpt = load_model_from_checkpoint(str(checkpoint_path), device=device)

    _, _, test_loader, _ = make_dataloaders(
        data_root=args.data_root,
        image_size=int(ckpt.get("image_size", args.image_size)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        subset_fraction=args.subset_fraction,
        eval_distortion=args.eval_distortion,
        blur_sigma=args.blur_sigma,
        noise_sigma=args.noise_sigma,
    )

    criterion = nn.CrossEntropyLoss()
    loss_weights = [args.w1, args.w2, args.w3, args.w4]

    threshold_sets = threshold_sets_from_args(args, ckpt.get("thresholds", [args.t1, args.t2, args.t3]))

    rows = []

    for th in threshold_sets:
        metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            thresholds=th,
            criterion=criterion,
            loss_weights=loss_weights,
            desc="Testing thresholds {}".format(th),
        )

        latency = benchmark_latency(
            model=model,
            loader=test_loader,
            device=device,
            thresholds=th,
            max_samples=args.latency_samples,
        )

        row = {
            "t1": th[0],
            "t2": th[1],
            "t3": th[2],
            "eval_distortion": args.eval_distortion,
            "blur_sigma": args.blur_sigma if args.eval_distortion == "blur" else "",
            "noise_sigma": args.noise_sigma if args.eval_distortion == "noise" else "",
            "policy_acc": metrics["policy_acc"],
            "exit1_acc": metrics["exit_acc"][0],
            "exit2_acc": metrics["exit_acc"][1],
            "exit3_acc": metrics["exit_acc"][2],
            "exit4_acc": metrics["exit_acc"][3],
            "avg_exit": metrics["avg_exit"],
            "avg_depth_cost_ratio": metrics["avg_depth_cost_ratio"],
            "exit1_count": metrics["exit_counts"][0],
            "exit2_count": metrics["exit_counts"][1],
            "exit3_count": metrics["exit_counts"][2],
            "exit4_count": metrics["exit_counts"][3],
            "lat_full_ms": latency["lat_full_ms"],
            "lat_early_ms": latency["lat_early_ms"],
            "speedup_x": latency["speedup_x"],
            "latency_samples": latency["latency_samples"],
            "exit_indices": ",".join(str(x) for x in model.exit_indices),
        }

        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    sweep_csv = paths["csv"] / "threshold_sweep.csv"
    df.to_csv(sweep_csv, index=False)

    plot_threshold_results(df, paths["plots"] / "threshold_accuracy_cost.png")
    plot_exit_distribution(df, paths["plots"] / "exit_distribution.png")

    best_idx = int(df["policy_acc"].idxmax())
    best_row = df.iloc[best_idx].to_dict()

    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "classes": len(classes),
            "exit_indices": list(model.exit_indices),
            "best_row_by_policy_acc": best_row,
            "all_threshold_results": rows,
        },
        paths["json"] / "test_metrics.json",
    )

    print("Saved:", sweep_csv)
    print("Saved:", paths["json"] / "test_metrics.json")


def plot_threshold_results(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return

    labels = ["{:.2f}/{:.2f}/{:.2f}".format(r.t1, r.t2, r.t3) for r in df.itertuples()]
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5))
    plt.plot(x, df["policy_acc"] * 100.0, marker="o", label="Policy Accuracy (%)")
    plt.plot(x, df["avg_depth_cost_ratio"] * 100.0, marker="s", label="Avg Cost Ratio (%)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.xlabel("Thresholds")
    plt.ylabel("Percent")
    plt.title("Threshold Sweep: Accuracy vs Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_exit_distribution(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return

    labels = ["{:.2f}/{:.2f}/{:.2f}".format(r.t1, r.t2, r.t3) for r in df.itertuples()]
    x = np.arange(len(labels))
    width = 0.20

    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * width, df["exit1_count"], width, label="Exit 1")
    plt.bar(x - 0.5 * width, df["exit2_count"], width, label="Exit 2")
    plt.bar(x + 0.5 * width, df["exit3_count"], width, label="Exit 3")
    plt.bar(x + 1.5 * width, df["exit4_count"], width, label="Final Exit")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.xlabel("Thresholds")
    plt.ylabel("Number of Samples")
    plt.title("Exit Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ==========================================================
# PART 10) Single-image inference
# What this part does:
#   Loads one image, applies the same preprocessing, predicts
#   the class, and reports which exit was used.
# ==========================================================

@torch.no_grad()
def run_infer(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError("Image not found: {}".format(image_path))

    device = get_device(args.cpu)

    model, classes, ckpt = load_model_from_checkpoint(str(checkpoint_path), device=device)

    image_size = int(ckpt.get("image_size", args.image_size))
    thresholds = [args.t1, args.t2, args.t3]

    if "thresholds" in ckpt and not args.override_thresholds:
        thresholds = ckpt["thresholds"]

    _, eval_tf = get_transforms(image_size=image_size)

    img = Image.open(str(image_path)).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)

    pred_idx, exit_num, confidence = model.predict_one_early_exit(x, thresholds)

    print("Using device:", device)
    print("Image:", image_path)
    print("Prediction:", classes[pred_idx])
    print("Exit used:", exit_num)
    print("Confidence: {:.4f}".format(confidence))
    print("Thresholds:", thresholds)
    print("Exit placement:", model.exit_indices)


# ==========================================================
# PART 11) Dataset checker
# What this part does:
#   Confirms that the dataset path is correct and prints
#   class count, image count, and per-class statistics.
# ==========================================================

def run_checkdata(args) -> None:
    data_root = Path(args.data_root)

    if not data_root.exists():
        raise FileNotFoundError("Dataset root not found: {}".format(data_root.resolve()))

    dataset = datasets.ImageFolder(str(data_root))

    counts = defaultdict(int)
    for _, y in dataset.samples:
        counts[int(y)] += 1

    values = list(counts.values())

    print("[OK] dataset_root:", data_root.resolve())
    print("[OK] classes:", len(dataset.classes))
    print("[OK] images:", len(dataset.samples))
    print("[OK] per-class count -> min:{}, max:{}, mean:{:.2f}".format(
        min(values),
        max(values),
        float(np.mean(values)),
    ))
    print("[OK] first class:", dataset.classes[0])
    print("[OK] last class: ", dataset.classes[-1])

    folder_count = len([
        p for p in data_root.iterdir()
        if p.is_dir() and re.match(r"^\d{3}\.", p.name)
    ])
    print("[OK] folders matching 000.name pattern:", folder_count)


# ==========================================================
# PART 12) Command-line interface
# What this part does:
#   Defines all terminal commands and their arguments.
# ==========================================================

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_root", type=str, default="computer-vision/256_ObjectCategories")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--subset_fraction", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--t1", type=float, default=0.95)
    p.add_argument("--t2", type=float, default=0.90)
    p.add_argument("--t3", type=float, default=0.85)

    p.add_argument("--w1", type=float, default=0.25)
    p.add_argument("--w2", type=float, default=0.50)
    p.add_argument("--w3", type=float, default=0.75)
    p.add_argument("--w4", type=float, default=1.00)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project 15 - Early Exit Image Classification with Branch Placement Grid Search"
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("checkdata")
    add_common_args(p_check)

    p_train = sub.add_parser("train")
    add_common_args(p_train)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--exit_indices", type=str, default="3,6,13")
    p_train.add_argument("--pretrained", action="store_true")
    p_train.add_argument("--freeze_backbone", action="store_true")
    p_train.add_argument("--lr_backbone", type=float, default=1.5e-4)
    p_train.add_argument("--lr_heads", type=float, default=1.0e-3)
    p_train.add_argument("--weight_decay", type=float, default=5e-4)
    p_train.add_argument("--label_smoothing", type=float, default=0.10)
    p_train.add_argument("--amp", action="store_true")
    p_train.add_argument("--early_stop_patience", type=int, default=0)

    p_grid = sub.add_parser("gridsearch")
    add_common_args(p_grid)
    p_grid.add_argument("--epochs", type=int, default=10)
    p_grid.add_argument("--placements", type=str, default="default")
    p_grid.add_argument("--pretrained", action="store_true")
    p_grid.add_argument("--freeze_backbone", action="store_true")
    p_grid.add_argument("--lr_backbone", type=float, default=1.5e-4)
    p_grid.add_argument("--lr_heads", type=float, default=1.0e-3)
    p_grid.add_argument("--weight_decay", type=float, default=5e-4)
    p_grid.add_argument("--label_smoothing", type=float, default=0.10)
    p_grid.add_argument("--amp", action="store_true")
    p_grid.add_argument("--early_stop_patience", type=int, default=0)

    p_test = sub.add_parser("test")
    add_common_args(p_test)
    p_test.add_argument("--checkpoint", type=str, default="")
    p_test.add_argument("--latency_samples", type=int, default=300)
    p_test.add_argument("--custom_thresholds", type=str, default="")
    p_test.add_argument("--eval_distortion", type=str, default="none", choices=["none", "blur", "noise"])
    p_test.add_argument("--blur_sigma", type=float, default=2.0)
    p_test.add_argument("--noise_sigma", type=float, default=10.0)

    p_infer = sub.add_parser("infer")
    add_common_args(p_infer)
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--image", type=str, required=True)
    p_infer.add_argument("--override_thresholds", action="store_true")

    return parser


# ==========================================================
# PART 13) Main program
# What this part does:
#   Runs the selected command from the terminal.
# ==========================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "checkdata":
        run_checkdata(args)

    elif args.cmd == "train":
        exit_indices = parse_exit_indices(args.exit_indices)
        train_model_for_placement(
            args=args,
            exit_indices=exit_indices,
            tag="main",
            save_as_main_best=True,
        )

    elif args.cmd == "gridsearch":
        run_gridsearch(args)

    elif args.cmd == "test":
        run_test(args)

    elif args.cmd == "infer":
        run_infer(args)

    else:
        raise ValueError("Unknown command: {}".format(args.cmd))


if __name__ == "__main__":
    main()