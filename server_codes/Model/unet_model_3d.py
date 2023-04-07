from .unet_parts import *


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3D, self).__init__()
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
        self.outc = OutConv(64, n_classes)

        self.deconv3d_1 = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)
        self.conv3d_1 = nn.Conv3d(32, 16, 1)
        self.deconv3d_2 = nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1)
        self.conv3d_2 = nn.Conv3d(8, 4, 1)
        self.deconv3d_3 = nn.ConvTranspose3d(4, 2, 4, stride=2, padding=1)
        self.conv3d_3 = nn.Conv3d(2, 1, 1)
        self.deconv3d_4 = nn.ConvTranspose3d(1, 1, 4, stride=2, padding=1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        B, C, H, W = x5.shape
        x_3d = x5.reshape([-1, 64, 16, H, W])

        x_3d = F.leaky_relu(self.deconv3d_1(x_3d))  # 32x32x32x32
        x_3d = F.leaky_relu(self.conv3d_1(x_3d))    # 16x32x32x32

        x_3d = F.leaky_relu(self.deconv3d_2(x_3d))  # 8x64x64x64
        x_3d = F.leaky_relu(self.conv3d_2(x_3d))    # 4x64x64x64

        x_3d = F.leaky_relu(self.deconv3d_3(x_3d))  # 2x128x128x128
        x_3d = F.leaky_relu(self.conv3d_3(x_3d))    # 1x128x128x128

        logits_3d = F.sigmoid(self.deconv3d_4(x_3d))  # 1x256x256x256

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, logits_3d
