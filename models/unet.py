import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Fusion2(nn.Module):
    def __init__(self, upper_channel_size: int, lower_channel_size: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up = double_conv(upper_channel_size + lower_channel_size, lower_channel_size)

    def forward(self, upper_feature_map: torch.Tensor, lower_feature_map: torch.Tensor) -> torch.Tensor:
        x = self.upsample(upper_feature_map)
        x = torch.cat([x, lower_feature_map], dim=1)
        x = self.dconv_up(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.fuse_up3 = Fusion2(512, 256)
        self.fuse_up2 = Fusion2(256, 128)
        self.fuse_up1 = Fusion2(128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # size=(N, 64, x.H, x.W)
        x = self.maxpool(conv1)  # size=(N, 64, x.H/2, x.W/2)

        conv2 = self.dconv_down2(x)  # size=(N, 128, x.H/2, x.W/2)
        x = self.maxpool(conv2)  # size=(N, 128, x.H/4, x.W/4)

        conv3 = self.dconv_down3(x)  # size=(N, 256, x.H/4, x.W/4)
        x = self.maxpool(conv3)  # size=(N, 256, x.H/8, x.W/8)

        conv4 = self.dconv_down4(x)  # size=(N, 512, x.H/8, x.W/8)

        x = self.fuse_up3(conv4, conv3)  # size=(N, 256, x.H/4, x.W/4)
        x = self.fuse_up2(x, conv2)  # size=(N, 128, x.H/2, x.W/2)
        x = self.fuse_up1(x, conv1)  # size=(N, 64, x.H, x.W)

        out = self.conv_last(x)  # size=(N, n_class, x.H, x.W)
        return out
