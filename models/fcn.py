import torch.nn as nn
from torchvision import models


class Fusion(nn.Module):
    def __init__(self, upper_channel_size, lower_channel_size):
        super().__init__()
        self.conv = nn.Conv2d(upper_channel_size, lower_channel_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, upper_feature_map, lower_feature_map):
        out = self.conv(upper_feature_map) + lower_feature_map
        out = self.relu(out)

        return out


class FCN4s(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)

        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fuse_fm4_and_fm3 = Fusion(512, 256)
        self.fuse_fm3_and_fm2 = Fusion(256, 128)
        self.fuse_fm2_and_fm1 = Fusion(128, 64)

        self.conv1k = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        fm1 = self.layer1(x)  # size=(N, 64, x.H/4, x.W/4)
        fm2 = self.layer2(fm1)  # size=(N, 128, x.H/8, x.W/8)
        fm3 = self.layer3(fm2)  # size=(N, 256, x.H/16, x.W/16)
        fm4 = self.layer4(fm3)  # size=(N, 512, x.H/32, x.W/32)

        up4 = self.upsample(fm4)  # size=(N, 512, x.H/16, x.W/16)
        fs43 = self.fuse_fm4_and_fm3(up4, fm3)  # size=(N, 256, x.H/16, x.W/16)

        up3 = self.upsample(fs43)  # size=(N, 256, x.H/8, x.W/8)
        fs32 = self.fuse_fm3_and_fm2(up3, fm2)  # size=(N, 128, x.H/8, x.W/8)

        up2 = self.upsample(fs32)  # size=(N, 128, x.H/4, x.W/4)
        fs21 = self.fuse_fm2_and_fm1(up2, fm1)  # size=(N, 64, x.H/4, x.W/4)

        up1 = self.upsample(fs21)  # size=(N, 64, x.H/2, x.W/2)
        up1 = self.upsample(up1)  # size=(N, 64, x.H, x.W)
        return self.conv1k(up1)
