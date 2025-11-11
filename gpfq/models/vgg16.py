import torch
import torch.nn as nn


class VGG16(nn.Module):
    """VGG-16 architecture implementation with batch normalization and configurable output classes."""

    def __init__(
        self,
        num_classes: int = 1000,
        input_channels: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.5,
        init_weights: bool = True,
    ):
        """Initialize VGG-16.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels (e.g., 3 for RGB).
            use_batch_norm: Whether to include batch normalization after conv layers.
            dropout: Dropout probability in the classifier.
            init_weights: Whether to apply custom weight initialization.
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        # VGG-16 configuration: 'M' denotes MaxPool2d
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

        self.features = self._make_layers(cfg, input_channels, use_batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg: list, input_channels: int, batch_norm: bool) -> nn.Sequential:
        """Construct feature extraction layers from configuration."""
        layers = []
        in_channels = input_channels

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming (He) initialization for conv/linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
