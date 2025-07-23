import torch
from models.UNET_regression import OutputConv, ConvReluBlock, SkipAndDownSample, UpSample
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=21, base_channels=64):
        super().__init__()
        self.grid_size = grid_size
        self.down1 = SkipAndDownSample(in_channels, base_channels)
        self.down2 = SkipAndDownSample(base_channels, base_channels*2)
        self.down3 = SkipAndDownSample(base_channels*2, base_channels*4)
        self.down4 = SkipAndDownSample(base_channels*4, base_channels*8)

        self.bottleneck = ConvReluBlock(base_channels*8, base_channels*16)

        self.up_1 = UpSample(base_channels*16, base_channels*8)
        self.up_2 = UpSample(base_channels*8, base_channels*4)
        self.up_3 = UpSample(base_channels*4, base_channels*2)
        self.up_4 = UpSample(base_channels*2, base_channels)
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x_down1, x_skip1 = self.down1(x)
        x_down2, x_skip2 = self.down2(x_down1)
        x_down3, x_skip3 = self.down3(x_down2)
        x_down4, x_skip4 = self.down4(x_down3)
        x_bottleneck = self.bottleneck(x_down4)
        x_up1 = self.up_1(x_bottleneck, x_skip4)
        x_up2 = self.up_2(x_up1, x_skip3)
        x_up3 = self.up_3(x_up2, x_skip2)
        x_up4 = self.up_4(x_up3, x_skip1)
        x_out = self.output(x_up4)
        return x_out
    def name(self):
        return "UNetRegressionFull"