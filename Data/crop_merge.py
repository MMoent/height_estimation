import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop(data_folder, img_id, x_crop_size, y_crop_size, x_interval, y_interval, x_crop_num, y_crop_num):
    img_path = os.path.join(data_folder, img_id + '_rgb.jp2')
    img = cv2.imread(img_path)
    for r in range(x_crop_num):
        for c in range(y_crop_num):
            if r == 0:
                if c == 0:
                    sub_im = img[0:x_crop_size, 0:y_crop_size, :]
                else:
                    sub_im = img[0:x_crop_size, c*(y_crop_size-y_interval):c*(y_crop_size-y_interval)+y_crop_size, :]
            else:
                if c == 0:
                    sub_im = img[r*(x_crop_size-x_interval):r*(x_crop_size-x_interval)+x_crop_size, 0:y_crop_size, :]
                else:
                    sub_im = img[r*(x_crop_size-x_interval):r*(x_crop_size-x_interval)+x_crop_size, c*(y_crop_size-y_interval):c*(y_crop_size-y_interval)+y_crop_size, :]
            cv2.imwrite(os.path.join('Test_cropped', img_id+'_'+str(r)+'_'+str(c)+'_rgb.png'), sub_im)


def merge(img_id, x_size, y_size, x_crop_size, y_crop_size, x_interval, y_interval, x_crop_num, y_crop_num):
    pred_merged = np.zeros((x_size, y_size))
    for r in range(x_crop_num):
        for c in range(y_crop_num):
            sub_im = cv2.imread(os.path.join('Pred_SIE_cropped', img_id+'_'+str(r)+'_'+str(c)+'_ndsm_pred.tif'), -1)
            if r == 0:
                if c == 0:
                    pred_merged[0:x_crop_size, 0:y_crop_size] = sub_im
                else:
                    pred_overlap = pred_merged[0:x_crop_size, c * (y_crop_size - y_interval):c * (y_crop_size - y_interval) + y_interval]
                    sub_overlap = sub_im[:, 0:y_interval]
                    overlap_area = (pred_overlap + sub_overlap) / 2
                    pred_merged[0:x_crop_size, c * (y_crop_size - y_interval):c * (y_crop_size - y_interval) + y_crop_size] = sub_im
                    pred_merged[0:x_crop_size, c * (y_crop_size - y_interval):c * (y_crop_size - y_interval) + y_interval] = overlap_area
            else:
                if c == 0:
                    overlap_area_1 = (pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_interval, 0:y_crop_size - y_interval] + sub_im[0:x_interval, 0:y_crop_size-y_interval]) / 2
                    overlap_area_2 = (pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_interval, y_crop_size-y_interval:y_crop_size] * 2 + sub_im[0:x_interval, y_crop_size-y_interval:y_crop_size]) / 3
                    pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_crop_size, 0:y_crop_size] = sub_im

                    pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_interval, 0:y_crop_size - y_interval] = overlap_area_1
                    pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_interval, y_crop_size - y_interval:y_crop_size] = overlap_area_2
                else:
                    overlap_area_1 = (pred_merged[r*(x_crop_size - x_interval):r*(x_crop_size-x_interval)+x_interval, c*(y_crop_size-y_interval):c*(y_crop_size-y_interval)+y_interval] * 3 + sub_im[0:x_interval:, 0:y_interval]) / 4
                    overlap_area_2 = (pred_merged[r*(x_crop_size - x_interval)+x_interval:r*(x_crop_size - x_interval)+x_crop_size, c*(y_crop_size-y_interval):c*(y_crop_size-y_interval)+y_interval] + sub_im[x_interval:, 0:y_interval]) / 2
                    overlap_area_3 = (pred_merged[r*(x_crop_size - x_interval):r*(x_crop_size - x_interval)+x_interval, c*(y_crop_size-y_interval)+y_interval:c*(y_crop_size-y_interval)+y_crop_size] + sub_im[0:x_interval, y_interval:]) / 2
                    pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_crop_size, c*(y_crop_size-y_interval):c*(y_crop_size-y_interval)+y_crop_size] = sub_im
                    pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_interval, c * (y_crop_size - y_interval):c * (y_crop_size - y_interval) + y_interval] = overlap_area_1
                    pred_merged[r * (x_crop_size - x_interval) + x_interval:r * (x_crop_size - x_interval) + x_crop_size, c * (y_crop_size - y_interval):c * (y_crop_size - y_interval) + y_interval] = overlap_area_2
                    pred_merged[r * (x_crop_size - x_interval):r * (x_crop_size - x_interval) + x_interval, c * (y_crop_size - y_interval) + y_interval:c * (y_crop_size - y_interval) + y_crop_size] = overlap_area_3

    pred_merged[pred_merged < 0.1] = 0
    plt.imshow(pred_merged)
    plt.show()
    cv2.imwrite(os.path.join('./Pred_SIE_merged', img_id+'_ndsm_pred.tif'), pred_merged)


def show_merged(img_id):
    img_path = os.path.join('./Pred_SIE_merged', img_id+'_ndsm_pred.tif')
    gt_path = os.path.join('./nDSM', img_id+'_ndsm.tif')
    img = cv2.imread(img_path, -1)
    gt = cv2.imread(gt_path, -1)
    rmse = np.sqrt(np.mean((img - gt)**2))
    vmin = min(np.min(gt), np.min(img))
    vmax = max(np.max(gt), np.max(img))
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth')
    plt.imshow(gt, vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2)
    plt.title(f'Prediction (RMSE: {rmse:.4f})')
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.savefig(os.path.join('Pred_SIE_GT_comparison', img_id+'_ret.png'), dpi=300)
    plt.show()


def main():
    data_folder = '/home/xiaomou/Codes/koeln/'
    test_img_ids = list(set([i[:8] for i in os.listdir('Test')]))
    x_size, y_size = 1000, 1000
    x_crop_size, y_crop_size = 256, 256
    x_crop_num, y_crop_num = 4, 4
    x_interval = 8
    y_interval = 8
    for img_id in test_img_ids:
        # crop(data_folder, img_id, x_crop_size, y_crop_size, x_interval, y_interval, x_crop_num, y_crop_num)
        # merge(img_id, x_size, y_size, x_crop_size, y_crop_size, x_interval, y_interval, x_crop_num, y_crop_num)
        show_merged(img_id)
        print(img_id, 'done')
        # c = input()


if __name__ == '__main__':
    main()
