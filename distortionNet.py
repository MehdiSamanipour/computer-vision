import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class DistortionNet(nn.Module):
    """
    Distortion classifier using Fourier magnitude spectrum as input.
    Classes: 0=pristine, 1=blur, 2=noise.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained_backbone: bool = True,
        input_mode: str = "spectrum",
    ):
        super().__init__()
        if input_mode not in ("spectrum", "rgb", "blend"):
            raise ValueError("input_mode must be one of: spectrum, rgb, blend")
        self.input_mode = input_mode
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d) and module is self.backbone.conv1:
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def rgb_to_spectrum(images: torch.Tensor) -> torch.Tensor:
        """
        Input Bx3xHxW in [0,1] -> output Bx1xHxW Fourier log-magnitude.
        """
        gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log1p(torch.abs(fft_shift))
        # Per-image normalization improves convergence for spectrum inputs.
        mean = magnitude.mean(dim=(2, 3), keepdim=True)
        std = magnitude.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        magnitude = (magnitude - mean) / std
        # Reuse pretrained RGB backbone by repeating single-channel spectrum.
        return magnitude.repeat(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "spectrum":
            model_input = self.rgb_to_spectrum(x)
        elif self.input_mode == "rgb":
            model_input = x
        else:
            spectrum = self.rgb_to_spectrum(x)
            model_input = 0.5 * x + 0.5 * spectrum
        return self.backbone(model_input)