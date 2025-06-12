import torch
from torch import nn
from torch.nn import functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        #input (B, C, H, W)
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.softmax = nn.Softmax(dim=2)
        self.scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_flat = self.flatten(x) # (B, C, H*W)
        x_flat_transpose = x_flat.transpose(1, 2) # (B, H*W, C)
        x_soft = self.softmax(torch.bmm(x_flat, x_flat_transpose)) # (B, C, C)
        x_new = torch.bmm(x_soft, x_flat) # (B, C, H*W)
        x_new_reshaped = torch.reshape(x_new, (b, c, h, w)) # (B, C, H*W) -> (B, C, H, W)
        x_out = x_new_reshaped * self.scale + x
        return x_out
    def name(self):
        return "ChannelAttentionModule"
    
class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        #input (B, C, H, W)
        super().__init__()
        inter = max(1, in_channels // 8)
        self.conv1 = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        B = self.flatten(self.conv1(x)) # (B, C, H*W)
        C = self.flatten(self.conv2(x)) # (B, C, H*W)
        D = self.flatten(self.conv3(x)) # (B, C, H*W)
        B_transpose = B.transpose(1, 2) # (B, H*W, C)
        BC  = torch.bmm(B_transpose, C) # (B, H*W, H*W)
        BC_soft = self.softmax(BC)
        A_new = torch.bmm(D, BC_soft) # (B, C, H*W)
        A_new_reshaped = torch.reshape(A_new, (b, c, h, w))
        A_out = A_new_reshaped * self.scale + x
        return A_out
    def name(self):
        return "PositionAttentionModule"
    
class ConvReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)
    def name(self):
        return "ConvReluBlock"
    
class DA_CNN(nn.Module):
    def __init__(self, in_channels, out_channels=1, grid_size=21, first_layer_filters=16, kernel_size=1):
        super().__init__()
        self.convcam1 = ConvReluBlock(in_channels, first_layer_filters)
        self.convcam2 = ConvReluBlock(first_layer_filters, first_layer_filters * 2)
        self.cam = ChannelAttentionModule(first_layer_filters * 2)
        self.convcam3 = ConvReluBlock(first_layer_filters * 2, first_layer_filters * 4)
        self.convpam1 = ConvReluBlock(in_channels, first_layer_filters)
        self.convpam2 = ConvReluBlock(first_layer_filters, first_layer_filters * 2)
        self.pam = PositionAttentionModule(first_layer_filters * 2)
        self.convpam3 = ConvReluBlock(first_layer_filters * 2, first_layer_filters * 4)
        fuse_conv_filters = first_layer_filters * 8
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.fuseconv = self.batch = nn.Sequential(
            nn.Conv2d(fuse_conv_filters, fuse_conv_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(fuse_conv_filters),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(fuse_conv_filters, out_channels)
    def forward(self, x):
        x_c1 = self.convcam1(x)
        x_c2 = self.convcam2(x_c1)
        x_cam = x_c2
        x_cam = self.cam(x_c2)
        x_c3 = self.convcam3(x_cam)
        x_p1 = self.convpam1(x)
        x_p2 = self.convpam2(x_p1)
        x_pam = x_p2
        x_pam = self.pam(x_p2)
        x_p3 = self.convpam3(x_pam)
        x_fuse = torch.cat((x_c3, x_p3), dim=1)
        x_fuse = self.fuseconv(x_fuse)
        x_pool = self.pool(x_fuse)
        x_flat = self.flatten(x_pool)
        x_out = self.linear(x_flat)
        return x_out
    def name(self):
        return "DA_CNN"
    


if __name__ == "__main__":
    import torch
    model = DA_CNN(6, 1)
    print(model)
    x = torch.randn(1, 6, 21, 21)
    y = model(x)
    print(y.shape)


            
