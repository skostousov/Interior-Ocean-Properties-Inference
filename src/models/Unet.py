import torch
import yaml
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.down1 = SkipAndDownSample(in_channels, 64)
        self.down2 = SkipAndDownSample(64, 128)
        self.down3 = SkipAndDownSample(128, 256)
        self.down4 = SkipAndDownSample(256, 512)

        self.bottleneck = ConvReluBlock(512, 1024)

        self.up_1 = UpSample(1024, 512)
        self.up_2 = UpSample(512, 256)
        self.up_3 = UpSample(256, 128)
        self.up_4 = UpSample(128, 64)
        self.output = Output(64, out_channels)
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


class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output_layer = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.output_layer(x)



class ConvReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class SkipAndDownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvReluBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    def forward(self, x):
        conv_output = self.conv(x)
        pooled_output = self.pool(conv_output)
        return pooled_output, conv_output
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2)
        self.conv = ConvReluBlock(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)