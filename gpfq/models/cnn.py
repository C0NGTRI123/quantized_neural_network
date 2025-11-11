import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network matching the architecture:
    2 × 32C3 → MP2 → 2 × 64C3 → MP2 → 2 × 128C3 → 128FC → 10FC
    """

    def __init__(
        self, input_channels: int = 3, num_classes: int = 10, dropout: float = 0.5, use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        # Build convolutional blocks: each block has 2 conv layers + optional BN + ReLU + (optional) pooling
        def conv_block(in_ch, out_ch, pool=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(input_channels, 32, pool=True),  # 2×32C3 → MP2
            conv_block(32, 64, pool=True),  # 2×64C3 → MP2
            conv_block(64, 128, pool=False),  # 2×128C3 (no pool)
        )

        # Classifier: 128FC → 10FC
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 8 * 8, 128),  # assuming input is 32x32 → after 2 pools: 32→16→8
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
