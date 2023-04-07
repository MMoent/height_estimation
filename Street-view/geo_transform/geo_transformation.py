import math
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


def generate_grid(h, w):
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xy = np.meshgrid(x, y)
    grid = torch.tensor([xy]).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid


def depth2voxel(img_depth, gsize):
    gsize = torch.tensor(gsize).int()
    n, c, h, w = img_depth.size()
    site_z = img_depth[:, 0, h // 2, w // 2] + 3.0
    voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize).cuda()

    # depth voxel
    grid_mask = generate_grid(gsize, gsize)
    grid_mask = grid_mask.expand(n, gsize, gsize, 2).cuda()
    grid_depth = torch.nn.functional.grid_sample(img_depth.cuda(), grid_mask, align_corners=True)
    voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
    voxel_depth = voxel_depth - voxel_sitez

    # occupancy voxel
    voxel_grid = torch.arange(-gsize / 2, gsize / 2, 1).float()
    voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize).cuda()
    voxel_ocupy = torch.ge(voxel_depth, voxel_grid).float().cpu()
    voxel_ocupy[:, gsize - 1, :, :] = 0
    voxel_ocupy = voxel_ocupy.cuda()

    # distance voxel
    voxel_dx = grid_mask[0, :, :, 0].view(1, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float() * float(
        gsize / 2.0)
    voxel_dx = voxel_dx.cuda()
    voxel_dy = grid_mask[0, :, :, 1].view(1, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float() * float(
        gsize / 2.0)
    voxel_dy = voxel_dy.cuda()
    voxel_dz = voxel_grid

    voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(voxel_dz)
    voxel_dis = voxel_dis.add(0.01)  # avoid 1/0 = nan
    voxel_dis = voxel_dis.mul(voxel_ocupy)
    voxel_dis = torch.sqrt(voxel_dis) - voxel_ocupy.add(-1.0).mul((gsize-1).float())

    return voxel_dis


def voxel2pano(voxel_dis, ori, pano_size):
    PI = math.pi
    r, c = pano_size[0], pano_size[1]
    n, s, t, tt = voxel_dis.size()
    k = s // 2
    # rays
    ori = ori.view(n, 1).expand(n, c).float()
    x = torch.arange(0, c, 1).float().view(1, c).expand(n, c)
    y = torch.arange(0, r, 1).float().view(1, r).expand(n, r)
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
    d_samples = torch.arange(0, float(k), 1).view(1, k, 1, 1).expand(n, k, r, c)
    samples_x = vx.mul(d_samples).add(k).long()
    samples_y = vy.mul(d_samples).add(k).long()
    samples_z = vz.mul(d_samples).add(k).long()
    samples_n = torch.arange(0, n, 1).view(n, 1, 1, 1).expand(n, k, r, c).long()
    samples_indices = samples_n.mul(s * s * s).add(samples_z.mul(s * s)).add(samples_y.mul(s)).add(samples_x)
    samples_indices = samples_indices.view(1, n * k * r * c)
    samples_indices = samples_indices[0, :].cuda()

    # get depth pano
    samples_depth = torch.index_select(voxel_dis, 1, samples_indices)
    samples_depth = samples_depth.view(n, k, r, c)
    min_depth = torch.min(samples_depth, 1)
    pano_depth = min_depth[0]
    pano_depth = pano_depth.view(n, 1, r, c)

    return pano_depth


def geo_projection(sate_depth, orientations, sate_gsd, pano_size, is_normalized=True):
    # recover the real depth
    if is_normalized:
        # sate_depth = sate_depth.mul(0.5).add(0.5).mul(255)
        sate_depth = sate_depth.mul(255)

    # step1: depth to voxel
    gsize = sate_depth.size()[3] * sate_gsd
    voxel_d = depth2voxel(sate_depth, gsize)

    # step2: voxel to panorama
    pano_depth = voxel2pano(voxel_d, orientations, pano_size)

    # step3: change pixel values of semantic panorama
    # pano_depth = pano_depth.mul(1.0/116.0)
    # pano_depth = pano_depth.add(-0.5).div(0.5)

    return pano_depth


def get_pano_by_id(im_id, out_dir, is_gt=False):
    pred_dir = '/home/xiaomou/Codes/height_estimation/Data/Pred_SIE_merged'
    gt_dir = '/home/xiaomou/Codes/height_estimation/Data/nDSM'
    im_path = os.path.join(
        gt_dir if is_gt else pred_dir,
        im_id + ('_ndsm.tif' if is_gt else '_ndsm_pred.tif')
    )
    sat_height = np.round(cv2.imread(im_path, -1)).astype(np.uint8)
    # enumerate available street view
    txt_file_name = 'reliable_metadata.txt'
    metadata = []
    with open(txt_file_name, 'r') as f:
        for line in f.readlines():
            one_line = eval(line)
            if one_line['im_id'] == im_id:
                metadata.append(one_line)
    for sample_meta in metadata:
        lat, lng = sample_meta['location']['lat'], sample_meta['location']['lng']
        sample_x, sample_y = sample_meta['coord']['x'], sample_meta['coord']['y']
        n = 256
        bottom, top, left, right = sample_x - n // 2, sample_x + n // 2, sample_y - n // 2, sample_y + n // 2
        if bottom < 0 or top >= 1000 or left < 0 or right >= 1000:
            continue
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # rect = plt.Rectangle((sample_x-n//2, sample_y-n//2), n, n, fill=False, edgecolor='red', linewidth=1)
        # ax.add_patch(rect)
        # plt.scatter(sample_x, sample_y, s=3, c='r')
        # plt.imshow(sat_height)
        # # plt.savefig(im_id + '_' + sample_meta['pano_id'] + '.png', dpi=300)
        # plt.show()
        print(sample_meta['pano_id'], lat, lng)
        sat_height_cropped = sat_height[left:right, bottom:top]     # im[y1:y2, x1:x2]
        sat_height_cropped = transforms.ToTensor()(sat_height_cropped)

        c, h, w = sat_height_cropped.size()
        sat_height_cropped = sat_height_cropped.view(1, c, h, w)
        orientations = torch.tensor(0 / 180.0 * math.pi)
        depth_pano = geo_projection(sat_height_cropped, orientations, 1, (n, 2*n))
        depth_pano = torch.squeeze(depth_pano).cpu().numpy()

        sat_height_show = torch.squeeze(sat_height_cropped).cpu().numpy()
        plt.subplot(1,2,1)
        plt.imshow(sat_height_show)
        plt.subplot(1,2,2)
        plt.imshow(depth_pano)
        plt.show()
        print("success")
        # pano_id_dir = os.path.join(out_dir, sample_meta['pano_id'])
        # if not os.path.exists(pano_id_dir):
        #     os.mkdir(pano_id_dir)
        #
        # equ = E2P.Equirectangular(depth_pano)
        # for i in [0, 90, 180, 270]:
        #     img_pers = equ.GetPerspective(90, i, 0, 256, 256)
        #     cv2.imwrite(os.path.join(pano_id_dir, 'perspective_'+str(i)+('_gt.png' if is_gt else '_pred.png')), img_pers)
        # cv2.imwrite(os.path.join(pano_id_dir, 'panorama' + ('_gt.png' if is_gt else '_pred.png')), depth_pano)


def main():
    im_id = '351_5645'
    out_dir = '/home/xiaomou/Codes/height_estimation/Street-view/geo_transform/result/' + im_id
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    get_pano_by_id(im_id, out_dir, True)


if __name__ == '__main__':
    main()
