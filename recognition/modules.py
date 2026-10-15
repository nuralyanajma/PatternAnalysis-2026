import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)

class ContextModule(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.Dropout3d(p=dropout),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )
 
    def forward(self, x):
        return x + self.block(x)

class UpsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvNormAct(in_ch, out_ch, kernel_size=3)

    def forward(self, x):
        return self.conv(self.up(x))


class LocalisationModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = ConvNormAct(in_ch, out_ch, kernel_size=3)
        self.conv1 = ConvNormAct(out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv1(self.conv3(x))
