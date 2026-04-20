# computer-vision

Early-exit deep neural networks for distorted images with distortion-aware expert branches and adaptive edge/cloud offloading.

Reference paper: "Early-exit deep neural networks for distorted images: providing an efficient edge offloading" ([arXiv:2108.09343](https://arxiv.org/abs/2108.09343)).

Repository: [https://github.com/MehdiSamanipour/computer-vision](https://github.com/MehdiSamanipour/computer-vision)

## Python version

This project is configured for Python `3.8.10` (`x64`).

## What is implemented

This codebase now follows the paper pipeline:

- Distortion types: Gaussian blur and Gaussian noise with paper levels.
- Distortion classifier (`DistortionNet`) that uses Fourier magnitude spectrum.
- MobileNetV2 early-exit architecture with 3 exits and expert branches:
  - `pristine` expert
  - `blur` expert
  - `noise` expert
- Two-stage training:
  - Train `Epristine` (backbone + pristine heads)
  - Fine-tune `Eblur` and `Enoise` (frozen backbone, train only expert heads)
- Adaptive inference:
  - detect distortion type
  - select expert branches
  - confidence-based early exit at edge, otherwise offload to cloud head

## Project files

- `MobileNetV2.py`: early-exit MobileNetV2 with expert heads and adaptive inference logic.
- `distortion_utils.py`: Gaussian blur/noise transformations and half-batch distortion policy.
- `distortionNet.py`: Fourier-spectrum distortion classifier model.
- `training_distortion_classifier.py`: training for distortion classifier.
- `train_early_exit_experts.py`: paper-style training for pristine/blur/noise experts.
- `infer_adaptive_offloading.py`: single-image adaptive inference and offloading decision.

## Dataset format

Use ImageFolder layout with pristine images for base classification training:

```text
dataset_root/
  class_000/
    img1.jpg
    img2.jpg
  class_001/
    ...
```

The code creates train/validation/test splits (80/10/10) internally.

## Setup

Install Python 3.8.10 and dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow opencv-python kagglehub
```

## Download Caltech256 to your path

Use the standalone downloader:

```bash
python dataset_download.py --target_dir "C:\your\dataset\path"
```

Equivalent core code:

```python
import kagglehub

path = kagglehub.dataset_download("jessicali9530/caltech256")
print("Path to dataset files:", path)
```

## 1) Train distortion classifier

```bash
python training_distortion_classifier.py ^
  --root_path "C:\path\to\dataset_root" ^
  --save_path "checkpoints/distortion_classifier.pt" ^
  --epochs 50 ^
  --classifier_input_mode spectrum
```

For maximum distortion-classifier accuracy on your setup, use:

```bash
python training_distortion_classifier.py ^
  --root_path "C:\path\to\dataset_root" ^
  --save_path "checkpoints/distortion_classifier_rgb.pt" ^
  --epochs 30 ^
  --batch_size 96 ^
  --classifier_input_mode rgb ^
  --disable_amp
```

Or auto-download before training:

```bash
python training_distortion_classifier.py ^
  --download_caltech256 ^
  --dataset_download_path "C:\your\dataset\path" ^
  --save_path "checkpoints/distortion_classifier.pt"
```

## 2) Train early-exit experts

```bash
python train_early_exit_experts.py ^
  --dataset_root "C:\path\to\dataset_root" ^
  --save_dir "checkpoints" ^
  --epochs_pristine 50 ^
  --epochs_expert 30
```

Or auto-download before training:

```bash
python train_early_exit_experts.py ^
  --download_caltech256 ^
  --dataset_download_path "C:\your\dataset\path" ^
  --save_dir "checkpoints"
```

`--num_classes` is optional. If omitted or set to `0`, the script infers the class count from `ImageFolder`.

Outputs:

- `checkpoints/early_exit_pristine.pt`
- `checkpoints/early_exit_blur.pt`
- `checkpoints/early_exit_noise.pt`
- `checkpoints/early_exit_all_experts.pt`

## 3) Run adaptive inference/offloading decision

```bash
python infer_adaptive_offloading.py ^
  --image_path "C:\path\to\image.jpg" ^
  --model_checkpoint "checkpoints/early_exit_all_experts.pt" ^
  --distortion_checkpoint "checkpoints/distortion_classifier.pt" ^
  --num_classes 257 ^
  --classifier_input_mode spectrum ^
  --target_confidence 0.8
```

Printed output includes:

- detected distortion type
- selected exit (`exit_1`, `exit_2`, `exit_3`, or `cloud`)
- predicted class id and confidence
- edge vs cloud offloading decision

## Notes

- The paper uses Caltech-256 (257 classes including clutter).
- Default confidence target is `0.8`.
- If no edge exit reaches target confidence, inference is offloaded to cloud head.
