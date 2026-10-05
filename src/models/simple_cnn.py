import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()

        self.features = nn.Sequential(
            # (B, 3, 224, 224) -> (B, 32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (B, 32, 112, 112) -> (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Giữ nguyên kích thước 56×56 ở tầng cuối
            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.max_pool = nn.AdaptiveMaxPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        avg_features = self.avg_pool(x)
        max_features = self.max_pool(x)

        x = torch.cat((avg_features, max_features), dim=1)
        return self.classifier(x)


def build_simple_cnn(num_classes=53):
    return SimpleCNN(num_classes=num_classes)