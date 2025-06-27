import torch
from torch import nn
from torch.nn import functional as F

class PixelWiseRegressor(nn.Module):
    def __init__(self, in_channels, out_channels=1, initial_conv_output=8, grid_size=21):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, initial_conv_output, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(initial_conv_output, initial_conv_output*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(initial_conv_output*2, initial_conv_output*4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(initial_conv_output*4, initial_conv_output*8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(initial_conv_output*8, initial_conv_output*4)
        self.linear_out = nn.Linear(initial_conv_output*4, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.linear_out(x)
        return x

    def name(self):
        return "PixelWiseRegressor"
    
if __name__ == "__main__":
    model = PixelWiseRegressor(in_channels=9, out_channels=1, grid_size=21)
    print(model)
    x = torch.randn(1, 9, 13, 13)  # Example input
    output = model(x)
    print(output.shape)  # Should be (1, 1) for a single pixel prediction
    print(output)
    print(model.name())  # Should print "PixelWiseRegressor"