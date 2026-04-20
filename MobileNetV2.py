import copy
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


EXPERT_TYPES = ("pristine", "blur", "noise")


class ExitHead(nn.Module):
    """Single exit point with an expert classifier per distortion type."""

    def __init__(self, in_channels: int, num_classes: int, expert_types: Iterable[str]):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.experts = nn.ModuleDict(
            {expert: nn.Linear(in_channels, num_classes) for expert in expert_types}
        )

    def forward(self, x: torch.Tensor, expert_type: str) -> torch.Tensor:
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.experts[expert_type](x)

    def copy_weights(self, from_expert: str, to_expert: str) -> None:
        self.experts[to_expert].load_state_dict(copy.deepcopy(self.experts[from_expert].state_dict()))


class MobileNetV2EarlyExitExperts(nn.Module):
    """
    MobileNetV2 backbone with 3 SPINN-style exits and distortion experts.
    Exit indices map to internal MobileNetV2 feature blocks.
    """

    def __init__(
        self,
        num_classes: int,
        expert_types: Iterable[str] = EXPERT_TYPES,
        exit_indices: Tuple[int, int, int] = (3, 6, 13),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=pretrained_backbone)
        self.features = backbone.features
        self.expert_types = tuple(expert_types)
        self.exit_indices = tuple(exit_indices)

        self.exit_channels = self._infer_exit_channels()
        self.exit_heads = nn.ModuleList(
            [ExitHead(ch, num_classes, self.expert_types) for ch in self.exit_channels]
        )
        self.final_head = ExitHead(1280, num_classes, self.expert_types)

        self.temperature = nn.Parameter(torch.ones(len(self.exit_heads) + 1), requires_grad=False)
        self._init_linear_layers()

    def _infer_exit_channels(self) -> List[int]:
        channels = []
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.exit_indices:
                    channels.append(x.size(1))
        return channels

    def _init_linear_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_expert_from(self, from_expert: str, to_expert: str) -> None:
        for exit_head in self.exit_heads:
            exit_head.copy_weights(from_expert, to_expert)
        self.final_head.copy_weights(from_expert, to_expert)

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def parameters_for_expert(self, expert_type: str):
        params = []
        for head in self.exit_heads:
            params.extend(list(head.experts[expert_type].parameters()))
        params.extend(list(self.final_head.experts[expert_type].parameters()))
        return params

    def forward(self, x: torch.Tensor, expert_type: str) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        exit_count = 0

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.exit_indices:
                outputs[f"exit_{exit_count + 1}"] = self.exit_heads[exit_count](x, expert_type)
                exit_count += 1

        outputs["cloud"] = self.final_head(x, expert_type)
        return outputs

    def calibrated_logits(self, logits: torch.Tensor, exit_id: int) -> torch.Tensor:
        return logits / self.temperature[exit_id]

    def adaptive_inference(
        self, x: torch.Tensor, expert_type: str, target_confidence: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        """
        Confidence-based exit decision.
        If no edge branch reaches threshold, result is returned from cloud head.
        """
        outputs = self.forward(x, expert_type)

        ordered_keys = [f"exit_{i}" for i in range(1, len(self.exit_heads) + 1)] + ["cloud"]
        best_key = ordered_keys[-1]
        best_conf = torch.zeros(x.size(0), device=x.device)
        best_logits = outputs["cloud"]

        for idx, key in enumerate(ordered_keys):
            logits = self.calibrated_logits(outputs[key], idx)
            probs = F.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
            if conf.mean().item() >= target_confidence:
                return {"selected_exit": key, "logits": logits, "confidence": conf}
            better_mask = conf > best_conf
            best_conf = torch.where(better_mask, conf, best_conf)
            best_logits = torch.where(
                better_mask.unsqueeze(1), logits, best_logits
            )
            best_key = key

        return {"selected_exit": best_key, "logits": best_logits, "confidence": best_conf}


def build_early_exit_mobilenet_v2(
    num_classes: int,
    pretrained_backbone: bool = True,
) -> MobileNetV2EarlyExitExperts:
    model = MobileNetV2EarlyExitExperts(
        num_classes=num_classes,
        expert_types=EXPERT_TYPES,
        exit_indices=(3, 6, 13),
        pretrained_backbone=pretrained_backbone,
    )
    return model