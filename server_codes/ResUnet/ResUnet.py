import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormRelu(nn.Module):
    def __init__(self, inc):
        super(BatchNormRelu, self).__init__()
        self.bn = nn.BatchNorm2d(inc)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super(ResidualBlock, self).__init__()
        """Conv Layer"""
        self.b1 = BatchNormRelu(inc)
        self.c1 = nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1)
        self.b2 = BatchNormRelu(outc)
        self.c2 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)

        """Identity Mapping"""
        self.s = nn.Conv2d(inc, outc, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip


class DecoderBlock(nn.Module):
    def __init__(self, inc, outc):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(inc, inc, kernel_size=2, stride=2)
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = ResidualBlock(inc+outc, outc)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        # print(x.shape, skip.shape)
        x = torch.cat([x, skip], dim=1)
        # print(x.shape)
        x = self.r(x)
        # print(x.shape)
        return x


class ResUnet(nn.Module):
    def __init__(self, inc1, outc):
        super(ResUnet, self).__init__()
        """ Encoder 1 """
        self.c11 = nn.Conv2d(inc1, 64, kernel_size=3, padding=1)
        self.br1 = BatchNormRelu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(inc1, 64, kernel_size=1, padding=0)
        """ Encoder 2, 3, 4 """
        self.r2 = ResidualBlock(64, 128, stride=2)
        self.r3 = ResidualBlock(128, 256, stride=2)
        self.r4 = ResidualBlock(256, 512, stride=2)
        """ Bridge """
        self.r5 = ResidualBlock(512, 1024, stride=2)
        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        """ Output """
        self.out = nn.Conv2d(64, outc, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs1):
        """ Encoder 1 """
        x = self.c11(inputs1)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs1)
        skip1 = x + s

        """ Encoder 2, 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)
        skip4 = self.r4(skip3)

        """ Bridge """
        b = self.r5(skip4)

        """ Decoder """
        d1 = self.d1(b, skip4)
        d2 = self.d2(d1, skip3)
        d3 = self.d3(d2, skip2)
        d4 = self.d4(d3, skip1)
        #
        # """ Output """
        output = self.out(d4)
        # output = self.sigmoid(output)

        return output


if __name__ == "__main__":
    x1 = torch.randn((2, 3, 256, 256))
    x2 = torch.randn((2, 12, 256, 256))

    model = ResUnet(3, 1)
    y = model(x1)
    print(y.shape)
