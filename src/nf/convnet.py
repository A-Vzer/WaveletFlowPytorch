import torch.nn as nn
from nf.layers import Conv2d, Conv2dZeros


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.cnn = self.st_net(in_channels, out_channels, hidden_channels)

    def st_net(self, in_channels, out_channels, hidden_channels):
        block = nn.Sequential(Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False),
                              Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)), nn.ReLU(inplace=False),
                              Conv2dZeros(hidden_channels, out_channels))
        return block

    def __call__(self, x):
        return self.cnn(x)