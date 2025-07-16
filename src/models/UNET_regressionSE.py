import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation

class UNetRegressionSE(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=17, base_filters=64, reduction=4):
        super().__init__()
        self.grid_size = grid_size
        self.down1 = SkipAndDownSample(in_channels, base_filters)
        self.se1   = SqueezeExcitation(base_filters, max(1, base_filters // reduction))
        self.down2 = SkipAndDownSample(base_filters, base_filters * 2)
        self.se2   = SqueezeExcitation(base_filters * 2, max(1, base_filters * 2 // reduction))
        self.down3 = SkipAndDownSample(base_filters * 2, base_filters * 4)
        self.se3   = SqueezeExcitation(base_filters * 4, max(1, base_filters * 4 // reduction))
        self.down4 = SkipAndDownSample(base_filters * 4, base_filters * 8)
        self.se4   = SqueezeExcitation(base_filters * 8, max(1, base_filters * 8 // reduction))
        self.bottleneck  = ConvReluBlock(base_filters * 8, base_filters * 16)
        self.se_bottleneck = SqueezeExcitation(base_filters * 16, max(1, base_filters * 16 // reduction))

        self.up_1  = UpSample(base_filters * 16, base_filters * 8)
        self.se_up1 = SqueezeExcitation(base_filters * 8, max(1, base_filters * 8 // reduction))
        self.up_2  = UpSample(base_filters * 8, base_filters * 4)
        self.se_up2 = SqueezeExcitation(base_filters * 4, max(1, base_filters * 4 // reduction))
        self.up_3  = UpSample(base_filters * 4, base_filters * 2)
        self.se_up3 = SqueezeExcitation(base_filters * 2, max(1, base_filters * 2 // reduction))
        self.up_4  = UpSample(base_filters * 2, base_filters)
        self.se_up4 = SqueezeExcitation(base_filters, max(1, base_filters // reduction))
        self.output = OutputConv(base_filters, out_channels, grid_size)
    def name(self):
        return "UNetRegressionSE"
    def forward(self, x):
        x_d1, x_skip1 = self.down1(x)
        x_d1 = self.se1(x_d1)
        x_d2, x_skip2 = self.down2(x_d1)
        x_d2 = self.se2(x_d2)
        x_d3, x_skip3 = self.down3(x_d2)
        x_d3 = self.se3(x_d3)
        x_d4, x_skip4 = self.down4(x_d3)
        x_d4 = self.se4(x_d4)
        x_b = self.bottleneck(x_d4)
        x_b = self.se_bottleneck(x_b)
        x_u1 = self.up_1(x_b,   x_skip4)
        x_u1 = self.se_up1(x_u1)
        x_u2 = self.up_2(x_u1,  x_skip3)
        x_u2 = self.se_up2(x_u2)
        x_u3 = self.up_3(x_u2,  x_skip2)
        x_u3 = self.se_up3(x_u3)
        x_u4 = self.up_4(x_u3,  x_skip1)
        x_u4 = self.se_up4(x_u4)
        return self.output(x_u4)

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
    new_grid_size = 17
    ds = TemporalDataset(transform=None, grid_size=new_grid_size)
    image, label = ds[0]
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    model = UNetRegressionSE(image.shape[0], 1, grid_size=new_grid_size).to(device)
    mu, std = ds.generate_mean_and_std([0])
    for i in range(image.shape[0]):
        image[i] = (image[i] - mu[i]) / std[i]
    # image_nm = (image - mu) / std
    image = image.unsqueeze(0)
    image = torch.rand(1, 6, new_grid_size, new_grid_size).to(device)
    new_input = image.to(device)
    output = model(new_input)
    print(f"model output: {output}")
    print(f"model output shape: {output.shape}, label shape: {label.shape}")
