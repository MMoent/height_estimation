import math
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from glob import glob
from torchvision import transforms
from equilib import equi2pers


def generate_grid(h, w):
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xy = np.meshgrid(x, y)
    grid = torch.from_numpy(np.array([xy])).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid


def depth2voxel(sate_occup, gsize):
    gsize = torch.tensor(gsize).int()
    n, _, c, h, w = sate_occup.size()

    # depth voxel
    grid_mask = generate_grid(gsize, gsize)
    grid_mask = grid_mask.cuda().expand(n, gsize, gsize, 2)

    # occupancy voxel
    voxel_grid = torch.arange(-gsize / 2, gsize / 2, 1).cuda().float()
    voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize)

    # t = torch.ge(voxel_depth, voxel_grid).float()

    sate_occup = torch.round(sate_occup.squeeze())

    # distance voxel
    voxel_dx = grid_mask[0, :, :, 0].view(1, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float() * float(
        gsize / 2.0)
    voxel_dy = grid_mask[0, :, :, 1].view(1, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float() * float(
        gsize / 2.0)
    voxel_dz = voxel_grid

    voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(voxel_dz)
    voxel_dis = voxel_dis.add(0.01)  # avoid 1/0 = nan
    voxel_dis = voxel_dis.mul(sate_occup)
    voxel_dis = torch.sqrt(voxel_dis) - sate_occup.add(-1.0).mul((gsize-1).float())

    return voxel_dis


def voxel2pano(voxel_dis, pano_size):
    PI = 3.1415926535
    r, c = pano_size[0], pano_size[1]
    n, s, t, tt = voxel_dis.size()
    k = s // 2
    # rays
    ori = torch.zeros(n, c).float().cuda()
    x = torch.arange(0, c, 1).float().cuda().view(1, c).expand(n, c)
    y = torch.arange(0, r, 1).float().cuda().view(1, r).expand(n, r)
    lon = x * 2 * PI / c + ori - PI
    lat = PI / 2.0 - y * PI / r
    sin_lat = torch.sin(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    cos_lat = torch.cos(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    sin_lon = torch.sin(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    cos_lon = torch.cos(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    vx = cos_lat.mul(sin_lon)
    vy = -cos_lat.mul(cos_lon)
    vz = sin_lat
    vx = vx.expand(n, k, r, c)
    vy = vy.expand(n, k, r, c)
    vz = vz.expand(n, k, r, c)
    #
    voxel_dis = voxel_dis.contiguous().view(1, n * s * s * s)

    # sample voxels along pano-rays
    d_samples = torch.arange(0, float(k), 1).cuda().view(1, k, 1, 1).expand(n, k, r, c)
    samples_x = vx.mul(d_samples).add(k).long()
    samples_y = vy.mul(d_samples).add(k).long()
    samples_z = vz.mul(d_samples).add(k).long()
    samples_n = torch.arange(0, n, 1).cuda().view(n, 1, 1, 1).expand(n, k, r, c).long()

    samples_indices = samples_n.mul(s * s * s).add(samples_z.mul(s * s)).add(samples_y.mul(s)).add(samples_x)
    samples_indices = samples_indices.view(1, n * k * r * c)
    samples_indices = samples_indices[0, :]

    # get depth pano
    samples_depth = torch.index_select(voxel_dis, 1, samples_indices)
    samples_depth = samples_depth.view(n, k, r, c)
    min_depth = torch.min(samples_depth, 1)
    pano_depth = min_depth[0]
    pano_depth = pano_depth.view(n, 1, r, c)

    return pano_depth


def get_pano_depth_3d(sate_occup, pano_size=(256, 512), sate_gsd=1):

    # step1: depth to voxel
    gsize = sate_occup.size()[3] * sate_gsd
    voxel_d = depth2voxel(sate_occup, gsize)

    # step2: voxel to panorama
    pano_depth = voxel2pano(voxel_d, pano_size)

    # step3: change pixel values of semantic panorama
    # pano_depth = pano_depth.mul(1.0/116.0)
    # pano_depth = pano_depth.div(255.0)
    pano_depth = torch.neg(pano_depth).add(128.)
    pano_depth = torch.where(pano_depth > 0, pano_depth, 0)
    pano_depth = pano_depth.div(128.)

    return pano_depth


if __name__ == '__main__':
    pass
