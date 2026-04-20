import random
from typing import Optional

import cv2
import numpy as np
import torch


BLUR_LEVELS = [1, 2, 3, 4, 5]
NOISE_LEVELS = [5, 10, 20, 30, 40]


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    # BCHW -> HWC float32 in [0, 1]
    image_np = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return np.clip(image_np, 0.0, 1.0)


def _to_tensor(image_np: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
    image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(reference.device)
    return tensor


def apply_gaussian_blur(image: torch.Tensor, sigma: int) -> torch.Tensor:
    image_np = _to_numpy(image)
    kernel_size = 4 * sigma + 1
    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return _to_tensor(blurred, image)


def apply_gaussian_noise(image: torch.Tensor, sigma: int) -> torch.Tensor:
    image_np = _to_numpy(image)
    noise_std = sigma / 255.0
    noise = np.random.normal(0.0, noise_std, size=image_np.shape).astype(np.float32)
    noisy = image_np + noise
    return _to_tensor(noisy, image)


def apply_distortion(
    image: torch.Tensor, distortion_type: str, sigma: Optional[int] = None
) -> torch.Tensor:
    if distortion_type == "pristine":
        return image
    if distortion_type == "blur":
        sigma = sigma if sigma is not None else random.choice(BLUR_LEVELS)
        return apply_gaussian_blur(image, sigma)
    if distortion_type == "noise":
        sigma = sigma if sigma is not None else random.choice(NOISE_LEVELS)
        return apply_gaussian_noise(image, sigma)
    raise ValueError("Unsupported distortion type: %s" % distortion_type)


def distort_half_batch(images: torch.Tensor, distortion_type: str) -> torch.Tensor:
    """
    Match the paper: in each mini-batch apply distortion to half of images.
    """
    if distortion_type == "pristine":
        return images
    out = images.clone()
    half = images.size(0) // 2
    for idx in range(half):
        out[idx] = apply_distortion(images[idx], distortion_type)
    return out
