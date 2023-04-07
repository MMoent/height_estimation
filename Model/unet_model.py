import torch

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.stup1 = StUp(1024, 512 // factor)
        self.stup2 = StUp(512, 256 // factor)
        self.stup3 = StUp(256, 128 // factor)
        self.stup4 = StUp(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x, x0):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        s = self.stup1(x5)
        s = self.stup2(s)
        s = self.stup3(s)
        s = self.stup4(s)
        logits_0, logits_90, logits_180, logits_270 = self.outc(s), self.outc(s), self.outc(s), self.outc(s)

        return logits, logits_0, logits_90, logits_180, logits_270
