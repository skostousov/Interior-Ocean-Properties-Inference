import torch
from torch import nn
from torch.nn import functional as F


class UNetRegression(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=21, first_out=32):
        super().__init__()
        self.grid_size = grid_size
        self.down1 = SkipAndDownSample(in_channels, first_out)
        self.down2 = SkipAndDownSample(first_out, first_out*2)
        self.down3 = SkipAndDownSample(first_out*2, first_out*4)
        self.down4 = SkipAndDownSample(first_out*4, first_out*8)

        self.bottleneck = ConvReluBlock(first_out*8, first_out*16)

        self.up_1 = UpSample(first_out*16, first_out*8)
        self.up_2 = UpSample(first_out*8, first_out*4, )
        self.up_3 = UpSample(first_out*4, first_out*2, )
        self.up_4 = UpSample(first_out*2, first_out, )
        self.output = OutputConv(first_out, out_channels, grid_size)
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

if __name__ == "__main__":  
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")   
    from utils.datasettemporal import TemporalDataset
    from utils.transforms import ToTensor, RescaledRotationTransform, Compose
    transform = Compose([ToTensor(), RescaledRotationTransform()])
    ds = TemporalDataset(transform=None)
    image, label = ds[0]
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    model = UNetRegression(image.shape[0], 1).to(device)
    mu, std = ds.generate_mean_and_std([0])
    for i in range(image.shape[0]):
        image[i] = (image[i] - mu[i]) / std[i]
    # image_nm = (image - mu) / std
    image = image.unsqueeze(0)
    image = torch.rand(1, 6, 21, 21).to(device)
    new_input = image.to(device)
    output = model(new_input)
    print(f"model output: {output}")
    print(f"model output shape: {output.shape}, label shape: {label.shape}")
