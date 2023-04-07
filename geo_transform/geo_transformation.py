import glob
import math
import os

import matplotlib
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchvision import transforms
from equilib import equi2pers


def generate_grid(h, w):
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xy = np.meshgrid(x, y)
    grid = torch.from_numpy(np.array([xy])).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid


def depth2voxel(img_depth, gsize):
    gsize = torch.tensor(gsize).int()
    n, c, h, w = img_depth.size()
    site_z = img_depth[:, 0, h // 2, w // 2] + 3.0
    voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize)

    # depth voxel
    grid_mask = generate_grid(gsize, gsize).cuda()
    grid_mask = grid_mask.expand(n, gsize, gsize, 2)
    grid_depth = torch.nn.functional.grid_sample(img_depth, grid_mask, align_corners=True)
    voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
    voxel_depth = voxel_depth - voxel_sitez

    # occupancy voxel
    voxel_grid = torch.arange(-gsize / 2, gsize / 2, 1).float().cuda()
    voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize)
    voxel_ocupy = torch.ge(voxel_depth, voxel_grid)  # no gradient

    # t = voxel_depth - voxel_grid
    # t[t > 0] = 1.0
    # t[t <= 0] = 0
    #
    # # t = torch.where(t > 0, 1., 0)
    # # t = voxel_depth.mul(voxel_ocupy.float())
    # # t = t.div(t)
    # # t = torch.nan_to_num(t)
    #
    # # voxel_ocupy[:, gsize - 1, :, :] = 0
    # # voxel_ocupy = voxel_ocupy.cuda()
    #
    # # distance voxel
    # voxel_dx = grid_mask[0, :, :, 0].view(1, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float() * float(
    #     gsize / 2.0)
    # voxel_dy = grid_mask[0, :, :, 1].view(1, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float() * float(
    #     gsize / 2.0)
    # voxel_dz = voxel_grid
    #
    # voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(voxel_dz)
    # voxel_dis = voxel_dis.add(0.01)  # avoid 1/0 = nan
    # voxel_dis = voxel_dis.mul(t)
    # voxel_dis = torch.sqrt(voxel_dis) - t.add(-1.0).mul((gsize-1).float())

    return voxel_ocupy


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


def get_pano_depth(sate_depth, pano_size=(256, 512), sate_gsd=1, is_normalized=False):
    # recover the real depth
    sate_depth = sate_depth.cuda()
    if is_normalized:
        # sate_depth = sate_depth.mul(0.5).add(0.5).mul(255)
        sate_depth = sate_depth.mul(255)

    # step1: depth to voxel
    gsize = sate_depth.size()[3] * sate_gsd
    voxel_d = depth2voxel(sate_depth, gsize)

    # step2: voxel to panorama
    pano_depth = voxel2pano(voxel_d, pano_size)

    # step3: change pixel values of semantic panorama
    # pano_depth = pano_depth.mul(1.0/116.0)
    # pano_depth = pano_depth.div(255.0)

    # pano_depth = torch.neg(pano_depth).add(128.)
    pano_depth = torch.where(pano_depth <= 128, pano_depth, 128)
    pano_depth = pano_depth.div(128.)

    # pano_depth = torch.reciprocal(pano_depth)
    return pano_depth


def main():
    data_dir = '../Data/street_sat'
    st_rgb_dir = '../Data/street_view_images'
    st_depth_dir = '../Data/street_view_depth'

    # matplotlib.use('Agg')
    total = glob.glob(os.path.join(data_dir, '*.tif'))
    for step, i in enumerate(total):
        sat_path = i
        s_id = os.path.split(i)[-1][:-4]

        # if os.path.exists(os.path.join('Perspective', s_id+'.png')):
        #     continue
        sat_height = torch.from_numpy(cv2.imread(sat_path, -1)).view(1, 1, 256, 256).cuda()
        voxel = depth2voxel(sat_height, 256).squeeze().permute(1, 2, 0).cpu().numpy()

        # save_path = os.path.join(data_dir, s_id + '.npy')
        # np.save(save_path, voxel)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(voxel)
        plt.axis("off")
        plt.savefig('voxel.png', dpi=300)
        plt.show()

        # t_sat = get_pano_depth(sat_height, (256, 512))

        # for h in range(4):
        #     rots = [{'roll': 0, 'pitch': 5 / 180 * torch.pi, 'yaw': h * 90 / 180 * torch.pi}]
        #     pers = equi2pers(t_sat, rots, 256, 256, 90, z_down=True)
        #     plt.subplot(1, 4, h+1)
        #     p = pers.squeeze().cpu().numpy()
        #     plt.imshow(p)
        #     plt.axis('off')

        # for h in range(4):
        #     st_rgb_path = os.path.join(st_rgb_dir, s_id + '_' + str(h * 90) + '.jpeg')
        #     st_depth_path = os.path.join(st_depth_dir, s_id + '_' + str(h * 90) + '.tif')
        #     if os.path.exists(st_depth_path):
        #         st_rgb = cv2.cvtColor(cv2.imread(st_rgb_path), cv2.COLOR_BGR2RGB)
        #         st_rgb = cv2.resize(st_rgb, (256, 256))
        #         st_depth = cv2.imread(st_depth_path, -1)
        #         # st_depth = (255 - st_depth).astype(np.uint8)
        #         # b = st_depth > 0
        #         # st_depth[b] -= st_depth[b].min()
        #         # st_depth[b] /= (st_depth[b].max()-st_depth[b].min())
        #         # st_depth = 1 - st_depth
        #         # st_depth[b] = 128 - st_depth[b]
        #         # st_depth *= 128
        #     else:
        #         st_rgb = np.zeros((256, 256, 3))
        #         st_depth = np.zeros((256, 256))
        #     plt.subplot(3, 4, h+5)
        #     plt.imshow(st_rgb)
        #     plt.axis('off')
        #     plt.subplot(3, 4, h+9)
        #     plt.imshow(st_depth)
        #     plt.axis('off')
        # # plt.savefig(os.path.join('Perspective', s_id+'.png'), dpi=300)
        # plt.show()
        # plt.close("all")

        # t_sat = t_sat.squeeze().cpu().numpy()
        # plt.imshow(t_sat)
        # # plt.axis('off')
        # plt.show()
        print(step, '/', len(total))


if __name__ == "__main__":
    main()
