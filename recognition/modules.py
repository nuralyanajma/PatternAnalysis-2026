import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock3D(nn.Module):
    def __init__(self, channels, dropout_rate=0.3):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = ConvBlock3D(channels, channels)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.conv2 = ConvBlock3D(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual