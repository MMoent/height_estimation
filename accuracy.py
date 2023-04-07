import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rmse(pred, gt):
    return np.sqrt(np.mean(np.square(pred-gt)))


def mae(pred, gt):
    return np.mean(np.abs(pred-gt))


def zncc(pred, gt):
    return np.mean((pred-np.mean(pred))*(gt-np.mean(gt))/(np.std(pred)*np.std(gt)))


def evaluate():
    matplotlib.use('Agg')

    dir_sat = './Data/street_sat'
    dir_no_street = './Experiments/NO_STREET_L1/test_output'
    dir_street_sil = './Experiments/1.0SAT_2.0ST_1.0SIE_0SSIM/test_output'
    dir_street_sil_ssim = './Experiments/1.0SAT_2.0ST_0.6SIE_0.6SSIM/test_output'

    im_ids = []
    with open('Data/test.txt') as f:
        for im_id in f:
            im_ids.append(im_id[:-1])

    for step, im_id in enumerate(im_ids):
        sat_rgb = cv2.cvtColor(cv2.imread(os.path.join(dir_sat, im_id+'.png')), cv2.COLOR_BGR2RGB)
        sat_gt = cv2.imread(os.path.join(dir_sat, im_id+'.tif'), -1)

        pred_no_street = cv2.imread(os.path.join(dir_no_street, im_id+'.tif'), -1)
        pred_street_sil = cv2.imread(os.path.join(dir_street_sil, im_id+'.tif'), -1)
        pred_street_sil_ssim = cv2.imread(os.path.join(dir_street_sil_ssim, im_id+'.tif'), -1)

        print(mae(pred_no_street, sat_gt), rmse(pred_no_street, sat_gt), zncc(pred_no_street, sat_gt))
        print(mae(pred_street_sil, sat_gt), rmse(pred_street_sil, sat_gt), zncc(pred_street_sil, sat_gt))
        print(mae(pred_street_sil_ssim, sat_gt), rmse(pred_street_sil_ssim, sat_gt), zncc(pred_street_sil_ssim, sat_gt))

        print(f'---------------------------{step+1}|{len(im_ids)}--------------------------------')

        vmin = min(pred_no_street.min(), pred_street_sil.min(), pred_street_sil_ssim.min())
        vmax = min(pred_no_street.max(), pred_street_sil.max(), pred_street_sil_ssim.max())

        plt.subplot(1, 5, 1)
        plt.imshow(sat_rgb)
        plt.xticks([])
        plt.yticks([])
        plt.title('RGB', fontsize=5)

        plt.subplot(1, 5, 2)
        plt.imshow(sat_gt, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title('GT', fontsize=5)

        plt.subplot(1, 5, 3)
        plt.imshow(pred_no_street, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title('w/o street-view', fontsize=5)

        plt.xlabel(f'MAE={mae(pred_no_street, sat_gt):.4f}\nRMSE={rmse(pred_no_street, sat_gt):.4f}', fontsize=5)

        plt.subplot(1, 5, 4)
        plt.imshow(pred_street_sil, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title('w/ street-view (SIL)', fontsize=5)
        plt.xlabel(f'MAE={mae(pred_street_sil, sat_gt):.4f}\nRMSE={rmse(pred_street_sil, sat_gt):.4f}', fontsize=5)

        plt.subplot(1, 5, 5)
        plt.imshow(pred_street_sil_ssim, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title('w/ street-view (SIL+SSIM)', fontsize=5)
        plt.xlabel(f'MAE={mae(pred_street_sil_ssim, sat_gt):.4f}\nRMSE={rmse(pred_street_sil_ssim, sat_gt):.4f}', fontsize=5)

        plt.savefig(os.path.join('./Data/test_output/Rinko', im_id+'.png'), dpi=300)
        plt.close('all')


if __name__ == "__main__":
    evaluate()
