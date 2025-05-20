import torch

import torch.nn as nn
import torch.nn.functional as F

class EBAM(nn.Module):
    """
    Enhanced Bottleneck Attention Module (EBAM) as described in the paper:
    "Attention Enhanced Deep Learning Model for Reconstruction and Downscaling of Thermocline Depth in the Tropical Indian Ocean"
    """
    def __init__(self, channels, reduction=16):
        super(EBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x

class CNN_EBAM(nn.Module):
    """
    CNN with EBAM attention for thermocline depth reconstruction/downscaling.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super(CNN_EBAM, self).__init__()
        layers = []
        prev_channels = in_channels
        for feat in features:
            layers.append(nn.Conv2d(prev_channels, feat, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feat))
            layers.append(nn.ReLU(inplace=True))
            layers.append(EBAM(feat))
            prev_channels = feat
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Conv2d(features[-1], features[-2], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-2], features[-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-3], out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example usage:
# model = CNN_EBAM(in_channels=1, out_channels=1)
# x = torch.randn(8, 1, 64, 64)
# out = model(x)