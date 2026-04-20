import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from MobileNetV2 import build_early_exit_mobilenet_v2
from distortionNet import DistortionNet


IDX_TO_DISTORTION = {0: "pristine", 1: "blur", 2: "noise"}


def load_image(path: str, input_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Adaptive offloading inference with expert branch selection.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_checkpoint", required=True, help="Checkpoint from train_early_exit_experts.py")
    parser.add_argument("--distortion_checkpoint", required=True, help="Checkpoint from training_distortion_classifier.py")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--target_confidence", type=float, default=0.8)
    parser.add_argument("--input_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distortion_model = DistortionNet(num_classes=3).to(device)
    distortion_ckpt = torch.load(args.distortion_checkpoint, map_location=device)
    distortion_model.load_state_dict(distortion_ckpt["model_state_dict"])
    distortion_model.eval()

    early_exit_model = build_early_exit_mobilenet_v2(
        num_classes=args.num_classes, pretrained_backbone=False
    ).to(device)
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    early_exit_model.load_state_dict(checkpoint["model_state_dict"])
    early_exit_model.eval()

    image = load_image(args.image_path, args.input_size).to(device)

    with torch.no_grad():
        distortion_logits = distortion_model(image)
        distortion_pred = torch.argmax(distortion_logits, dim=1).item()
        selected_expert = IDX_TO_DISTORTION[distortion_pred]

        result = early_exit_model.adaptive_inference(
            image, expert_type=selected_expert, target_confidence=args.target_confidence
        )

        probs = torch.softmax(result["logits"], dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        conf = torch.max(probs, dim=1).values.item()

    print("Detected distortion : %s" % selected_expert)
    print("Selected exit       : %s" % result["selected_exit"])
    print("Predicted class id  : %d" % pred_class)
    print("Prediction confidence: %.4f" % conf)

    if result["selected_exit"] == "cloud":
        print("Offloading decision : OFFLOAD to cloud")
    else:
        print("Offloading decision : INFER on edge")


if __name__ == "__main__":
    main()
