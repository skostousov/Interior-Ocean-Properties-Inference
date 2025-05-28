import torch
import torch.nn as nn
import math
from netCDF4 import Dataset
from tqdm.notebook import tqdm as tqdm
import torch.utils.checkpoint
from torch.optim.optimizer import Optimizer, required
import torch
from einops import rearrange
from torch import nn



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Using average and max pooling to capture spatial features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass to compute channel-wise attention
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Self-attention mechanism to capture long-range dependencies in spatial dimensions
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # Forward pass to compute the attention map and apply it to the input tensor
        b, c, h, w = x.shape


        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # [B, head, C/head, HW] * [B, head, HW, C/head] * [head, 1, 1] ==> [B, head, C/head, C/head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [B, head, C/head, C/head] * [B, head, C/head, HW] ==> [B, head, C/head, HW]
        out = (attn @ v)

        # [B, head, C/head, HW] ==> [B, head, C/head, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# Spatial attention mechanism to focus on important spatial locations
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Convolutional layer to compute spatial attention based on average and max pooling
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass to compute spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# EBAM (Enhanced Block Attention Module) that combines channel and spatial attention mechanisms
class EBAM(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True, kernel_size=7):
        super(EBAM, self).__init__()
        self.ca = Attention(dim, num_heads, bias)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# CNN model with embedded attention mechanisms
class EBAM_CNN(nn.Module):
    def __init__(self):
        super(EBAM_CNN, self).__init__()
        # Define the number of channels at each layer
        self.channels_num_1 = 6
        self.channels_num_2 = 32
        self.channels_num_3 = 64
        self.channels_num_4 = 128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ELU()
        # Define EBAM blocks for each feature map size
        self.ebam_1 = EBAM(self.channels_num_1)
        self.ebam_2 = EBAM(self.channels_num_2)
        self.ebam_3 = EBAM(self.channels_num_3)
        self.ebam_4= EBAM(self.channels_num_4)
        
        # Define convolutional layers to process the input
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride = 1, padding=0),
            nn.BatchNorm2d(32),
            #nn.Dropout(0.1)         
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride = 1,padding=0),
            nn.BatchNorm2d(64),
            #nn.Dropout(0.1)  
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, 3,stride = 1,padding=0),
            nn.BatchNorm2d(128),
            #nn.Dropout(0.1)
             
        )

        # Fully connected layers for final prediction
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.1),  
            nn.ELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.1), 
            #nn.LeakyReLU()
            nn.ELU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32,1)
#             nn.Dropout(0.5),
#             nn.ELU()     
        )

    def forward(self, x):
        # Forward pass through the network with attention applied
        # print(x)
        out = self.ebam_1(x)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.ebam_2(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.ebam_3(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.ebam_4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # out = self.fc4(out)
        return out

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")   
    from utils.datasettemporal import TemporalDataset
    













