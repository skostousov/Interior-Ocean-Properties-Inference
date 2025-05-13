import torch
from torch import nn
from torch.nn import functional as F

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
        self.output = OutputConv(64, out_channels)
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

class OutputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.output_layer(x)

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

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = UNet(13, 1).to(device)

    input = torch.rand(1, 13, 20, 20).to(device)
    output = model(input)
    print(output.shape)
