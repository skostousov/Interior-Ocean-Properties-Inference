import torch
from torch import nn
from torch.nn import functional as F


class UNetRegression(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=21, first_out=64):
        super().__init__()
        self.grid_size = grid_size
        self.down1 = SkipAndDownSample(in_channels, first_out)
        self.down2 = SkipAndDownSample(first_out, first_out*2)
        self.down3 = SkipAndDownSample(first_out*2, first_out*4)
        self.bottleneck = ConvReluBlock(first_out*4, first_out*8)

        self.up_1 = UpSample(first_out*8, first_out*4)
        self.up_2 = UpSample(first_out*4, first_out*2, )
        self.up_3 = UpSample(first_out*2, first_out, )
        self.output = OutputConv(first_out, out_channels, grid_size)
    def forward(self, x):
        x_down1, x_skip1 = self.down1(x)
        x_down2, x_skip2 = self.down2(x_down1)
        x_down3, x_skip3 = self.down3(x_down2)
        x_bottleneck = self.bottleneck(x_down3)
        x_up1 = self.up_1(x_bottleneck, x_skip3)
        x_up2 = self.up_2(x_up1, x_skip2)
        x_up3 = self.up_3(x_up2, x_skip1)
        x_out = self.output(x_up3)
        return x_out
    def name(self):
        return "UNetRegression"

class OutputConv(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size):
        super().__init__()
        # self.output = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.Flatten(),
        #     nn.Linear(out_channels * grid_size * grid_size, 1),
        #     nn.ReLU(inplace=1True)
        # )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
    def forward(self, x):
        # return self.output(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)
class ConvReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class SkipAndDownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvReluBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        conv_output = self.conv(x)
        pooled_output = self.pool(conv_output)
        return pooled_output, conv_output
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2)
        self.conv = ConvReluBlock(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[:2] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:],
                               mode = 'bilinear', align_corners=False)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class SimplePixelRegressor(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        # conv blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # → 32×13×13
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # → 64×13×13
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# → 128×13×13

        # global pool to collapse 13×13 → 1×1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # final regressor
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # shape: (B,32,13,13)
        x = F.relu(self.conv2(x))    # shape: (B,64,13,13)
        x = F.relu(self.conv3(x))    # shape: (B,128,13,13)

        x = self.global_pool(x)      # shape: (B,128,1,1)
        x = x.view(x.size(0), -1)    # shape: (B,128)
        x = self.fc(x)               # shape: (B,1)
        return x          # shape: (B,)

if __name__ == "__main__":
    model = SimplePixelRegressor(in_channels=9, out_channels=1)
    print(model)
    x = torch.randn(1, 9, 13, 13)  # Example input
    output = model(x)
    print(output.shape)  # Should be (1, 1, 13, 13) for a single pixel prediction
    print(output)