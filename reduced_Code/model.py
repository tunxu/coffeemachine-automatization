import torch.nn as nn

class DSCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = nn.Sequential(
            # Standard conv first
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Depthwise separable blocks
            self._ds_block(32, 32),
            self._ds_block(32, 32),
            self._ds_block(32, 32),
            self._ds_block(32, 32),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def _ds_block(self, in_ch, out_ch):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)