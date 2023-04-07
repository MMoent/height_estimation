import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from Model import UNet3D


def get_resnet50():
    model = torchvision.models.resnet50(pretrained=True)
    feature = nn.Sequential(*list(model.children())[:-2])
    feature[7][0].conv2.stride = (1, 1)
    feature[7][0].downsample[0].stride = (1, 1)
    return feature


class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.feature_extraction = get_resnet50()
        self.conv3d_1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.conv3d_2 = nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1)

    def forward(self, img):
        z_2d = self.feature_extraction(img)
        B,C,H,W = z_2d.shape
        z_3d = z_2d.reshape([-1, 256, 8, H, W])
        z_3d = F.leaky_relu(self.conv3d_1(z_3d))
        z_3d = F.leaky_relu(self.conv3d_2(z_3d))
        return z_3d


if __name__ == "__main__":
    encoder = Encoder3D()
    # print(encoder)
    x = torch.randn((2, 3, 256, 256))
    model = UNet3D(3, 1)

    y = model(x)
    print(y.shape)
