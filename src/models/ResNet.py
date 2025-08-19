import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="in", use_dropout=False):
        super().__init__()
        norm_layer = nn.InstanceNorm2d if norm == "in" else nn.BatchNorm2d
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class ResNetImageToImage(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=6, base_channels=64, norm="in", final_activation=None, grid_size=None):
        super().__init__()
        norm_layer = nn.InstanceNorm2d if norm == "in" else nn.BatchNorm2d
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(base_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(base_channels, norm=norm))
        layers += [
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=True),
        ]
        if final_activation is not None:
            layers.append(final_activation)
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        B = x.size(0)
        x = self.model(x)
        return x

    def name(self):
        return "ResNetImageToImage"


class ResNetImageToValue(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=6, base_channels=64, norm="in", final_activation=None, grid_size=None):
        super().__init__()
        norm_layer = nn.InstanceNorm2d if norm == "in" else nn.BatchNorm2d
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(base_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(base_channels, norm=norm))
        layers += [
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=True),
        ]
        if final_activation is not None:
            layers.append(final_activation)
        self.model = nn.Sequential(*layers)
        self.output = nn.AdaptiveMaxPool2d((1, 1))


    def forward(self, x):
        B = x.size(0)
        x = self.model(x)
        x = self.output(x)
        x = x.view(B, 1)
        return x

    def name(self):
        return "ResNetImageToValue"

if __name__ == "__main__":
    # Example: 5-channel input -> 1-channel output, preserves HxW
    net = ResNetImageToImage(in_channels=5, out_channels=1, n_blocks=6, base_channels=64, norm="in", final_activation=None)
    x = torch.randn(64, 5, 13, 13)
    y = net(x)
    print(x.shape, "->", y.shape)  # torch.Size([64, 5, 9, 9]) -> torch.Size([64, 1, 9, 9])

    res_block = ResidualBlock(channels=64, norm="in", use_dropout=True)
    x = torch.randn(64, 64, 9, 9)
    y = res_block(x)
    print(x.shape, "->", y.shape)  # torch.Size([64, 64, 9, 9]) -> torch.Size([64, 64, 9, 9])
