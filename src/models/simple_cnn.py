import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_simple_cnn(num_classes=53):
    return SimpleCNN(num_classes=num_classes)