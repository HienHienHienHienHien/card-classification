import torch.nn as nn
from torchvision import models




class TransferCNN(nn.Module):
    """EfficientNet-B0 pretrained trên ImageNet, thay head 1000 lớp -> num_classes."""


    def __init__(self, num_classes=53, pretrained=True, dropout=0.3):
        super().__init__()


        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)


        self.features = backbone.features
        self.avgpool = backbone.avgpool


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )


    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)




def build_transfer_cnn(num_classes=53, pretrained=True):
    return TransferCNN(num_classes=num_classes, pretrained=pretrained)