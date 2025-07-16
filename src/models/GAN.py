import torch.nn.utils.spectral_norm as SN
import torch.nn as nn
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation

class PatchDiscriminatorConditional(nn.Module):
    def __init__(self, in_channels=8, base_filters=64):
        super(PatchDiscriminatorConditional, self).__init__()
        self.layer1 = self.disc_block(in_channels, base_filters, normalize=False)
        self.layer2 = self.disc_block(base_filters, base_filters * 2)
        self.layer3 = self.disc_block(base_filters * 2, base_filters * 4)
        # self.layer4 = self.disc_block(base_filters * 4, base_filters * 8)
        self.final = SN(nn.Conv2d(base_filters * 4, 1, kernel_size=4, stride=1, padding=0, bias=False))
    def disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [SN(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    def name(self):
        return "PatchDiscriminatorConditional"
    def forward(self, inp, mld):
        x = torch.cat((inp, mld), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.final(x)
        return x

class GeneratorUNetRegressionSEConditional(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=13, base_filters=64, reduction=4):
        super().__init__()
        self.grid_size = grid_size
        self.down1 = SkipAndDownSample(in_channels, base_filters)
        self.se1   = SqueezeExcitation(base_filters, max(1, base_filters // reduction))
        self.down2 = SkipAndDownSample(base_filters, base_filters * 2)
        self.se2   = SqueezeExcitation(base_filters * 2, max(1, base_filters * 2 // reduction))
        self.down3 = SkipAndDownSample(base_filters * 2, base_filters * 4)
        self.se3   = SqueezeExcitation(base_filters * 4, max(1, base_filters * 4 // reduction))
        self.bottleneck  = ConvReluBlock(base_filters * 4, base_filters * 8)
        self.se_bottleneck = SqueezeExcitation(base_filters * 8, max(1, base_filters * 8 // reduction))

        self.up_1  = UpSample(base_filters * 8, base_filters * 4)
        self.se_up1 = SqueezeExcitation(base_filters * 4, max(1, base_filters * 4 // reduction))
        self.up_2  = UpSample(base_filters * 4, base_filters * 2)
        self.se_up2 = SqueezeExcitation(base_filters * 2, max(1, base_filters * 2 // reduction))
        self.up_3  = UpSample(base_filters * 2, base_filters)
        self.se_up3 = SqueezeExcitation(base_filters,     max(1, base_filters // reduction))
        self.output = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    def name(self):
        return "UNetRegressionSE"
    def forward(self, x):
        x_d1, x_skip1 = self.down1(x)
        x_d1 = self.se1(x_d1)
        x_d2, x_skip2 = self.down2(x_d1)
        x_d2 = self.se2(x_d2)
        x_d3, x_skip3 = self.down3(x_d2)
        x_d3 = self.se3(x_d3)
        x_b = self.bottleneck(x_d3)
        x_b = self.se_bottleneck(x_b)
        x_u1 = self.up_1(x_b, x_skip3)
        x_u1 = self.se_up1(x_u1)
        x_u2 = self.up_2(x_u1, x_skip2)
        x_u2 = self.se_up2(x_u2)
        x_u3 = self.up_3(x_u2, x_skip1)
        x_u3 = self.se_up3(x_u3)
        return self.output(x_u3)
    
class GeneratorUNetRegressionSEConditional2(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=13, base_filters=64, reduction=4):
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
        self.output = nn.Conv2d(base_filters, out_channels, kernel_size=1)
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
        x_u1 = self.up_1(x_b, x_skip4)
        x_u1 = self.se_up1(x_u1)
        x_u2 = self.up_2(x_u1, x_skip3)
        x_u2 = self.se_up2(x_u2)
        x_u3 = self.up_3(x_u2, x_skip2)
        x_u3 = self.se_up3(x_u3)
        x_u4 = self.up_4(x_u3, x_skip1)
        x_u4 = self.se_up4(x_u4)
        return self.output(x_u4)

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
        x = self.conv(x)
        return x
    
    
class PatchDiscriminatorRegressionRandom(PatchDiscriminatorConditional):
    def __init__(self, in_channels=1, base_filters=64):
        super(PatchDiscriminatorRegressionRandom, self).__init__(in_channels, base_filters)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)
        return x
    
class GeneratorUNetRegressionRandom(nn.Module):
    def __init__(self, z_dim=100, out_channels=1, base_filters=64, reduction=4):
        super().__init__()
        self.z_dim = z_dim
        self.base_filters = base_filters

        self.fc = nn.Linear(z_dim, base_filters * 4 *16*16)
        self.bottleneck = ConvReluBlock(base_filters * 4, base_filters * 8)
        self.se_bottleneck = SqueezeExcitation(base_filters * 8, max(1, base_filters * 8 // reduction))
        self.up_1 = UpSample(base_filters * 8, base_filters * 4)
        self.se_up1 = SqueezeExcitation(base_filters * 4, max(1, base_filters * 4 // reduction))
        self.up_2 = UpSample(base_filters * 4, base_filters * 2)
        self.se_up2 = SqueezeExcitation(base_filters * 2, max(1, base_filters * 2 // reduction))
        self.up_3 = UpSample(base_filters * 2, base_filters)
        self.se_up3 = SqueezeExcitation(base_filters, max(1, base_filters // reduction))
        self.output = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    def name(self):
        return "UNetRegressionRandom"
    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, self.base_filters * 4, 16, 16) #(B, z_dim) -> (B, base_filters * 4 * 16 * 16) -> (B, base_filters * 4, 16, 16)

        x_b = self.bottleneck(x) #(B, base_filters * 4, 16, 16) -> (B, base_filters * 8, 16, 16)
        x_b = self.se_bottleneck(x_b) 

        dummy_skip = torch.zeros_like(x) # (B, base_filters * 4, 16, 16)
        x_u1 = self.up_1(x_b, dummy_skip) # 
        x_u1 = self.se_up1(x_u1)

        dummy_skip = torch.zeros_like(x_u1)
        x_u2 = self.up_2(x_u1, dummy_skip)
        x_u2 = self.se_up2(x_u2)

        dummy_skip = torch.zeros_like(x_u2)
        x_u3 = self.up_3(x_u2, dummy_skip)
        x_u3 = self.se_up3(x_u3)

        return self.output(x_u3)
    
class DCGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_filters=64):
        super(DCGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )
    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)  

class DCGANGenerator(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, base_filters=64):
        super(DCGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_filters * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    
    # inp = torch.randn(8, 7, 128, 128)  # Example input
    # mld = torch.randn(8, 1, 128, 128)  # Example mld input
    # D_output = D(inp, mld)
    # G_output = G(inp)
    # print(f"Discriminator output: {D_output.shape}")  # Should be (batch_size, 1, 1, 1) after the final conv layer
    # print(f"Generator output: {G_output.shape}")  # Should be (batch_size, 1, 128, 128) after the final conv layer
    # print(f"D of G output: {D(inp, G_output).shape}")  # Should be (batch_size, 1, 1, 1) after the final conv layer
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    # from data.argo.alternate_dataset import myDataset
    # dataset = myDataset(season="autumn")
    # print(f"Dataset length: {len(dataset)}")
    # X, y = dataset.return_full_dataset()
    from utils.datasettemporal import TemporalDataset
    temporal_dataset = TemporalDataset()
    inp, mld = temporal_dataset.feature_map, temporal_dataset.annotations_map
    inp, mld = torch.tensor(inp, dtype=torch.float32), torch.tensor(mld, dtype=torch.float32)
    print(f"Input shape: {inp.shape}, Target shape: {mld.shape}")
    D = PatchDiscriminatorConditional(in_channels=inp.shape[1]+mld.shape[1], )
    G = GeneratorUNetRegressionSEConditional(in_channels=inp.shape[1], out_channels=1, grid_size=128, base_filters=64, reduction=4)
    D_output = D(inp, mld)
    G_output = G(inp)
    print(f"Discriminator output: {D_output.shape}")  # Should be (batch_size, 1, 1, 1) after the final conv layer
    print(f"Generator output: {G_output.shape}")  # Should be (batch_size, 1, 128, 128) after the final conv layer
    print(f"D of G output: {D(inp, G_output).shape}")  # Should be (batch_size, 1, 1, 1) after the final conv layer