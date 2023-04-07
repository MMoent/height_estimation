import os
import glob
import random
import numpy as np
import cv2


def create_dataset():
    data_folder = '/home/xiaomou/Codes/koeln'
    dtm_folder = './DTM'
    img_ids = [i[:-12] for i in os.listdir(dtm_folder)]
    random.shuffle(img_ids)
    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
    train_ids = img_ids[:int(len(img_ids)*train_ratio)]
    test_ids = img_ids[int(len(img_ids)*train_ratio):int(len(img_ids)*(train_ratio+test_ratio))]

    for id in img_ids:
        rgb = cv2.imread(os.path.join(data_folder, id+'_rgb.jp2'))
        dsm = cv2.imread(os.path.join(data_folder, id+'_dem.tif'), -1)
        dtm = cv2.imread(os.path.join(dtm_folder, id+'_dem_dtm.tif'), -1)
        ndsm = dsm - dtm
        ndsm[ndsm < 0.1] = 0
        cv2.imwrite(os.path.join('./nDSM', id+'_ndsm.tif'), ndsm)
        target_dir = 'Train' if id in train_ids else ('Test' if id in test_ids else 'Val')
        interval = 250
        for r in range(4):
            for c in range(4):
                sub_rbg = rgb[r * interval:(r + 1) * interval, c * interval:(c + 1) * interval, :]
                sub_ndsm = ndsm[r * interval:(r + 1) * interval, c * interval:(c + 1) * interval]
                cv2.imwrite(os.path.join(target_dir, id+'_'+str(r)+'_'+str(c)+'_rgb.png'), sub_rbg)
                cv2.imwrite(os.path.join(target_dir, id+'_'+str(r)+'_'+str(c)+'_ndsm.tif'), sub_ndsm)
        print(id, 'done')


def cal_mean_and_std():
    data_folder = '/home/xiaomou/Codes/height_estimation/Data/street_view_images'
    mean, std = np.array([.0, .0, .0]), np.array([.0, .0, .0])

    for step, i in enumerate(os.listdir(data_folder)):
        path = os.path.join(data_folder, i)
        raw_img = cv2.imread(path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img = np.zeros_like(raw_img, dtype=np.float32)
        img = raw_img / 255.0
        mean[0] += np.mean(img[:, :, 0])
        mean[1] += np.mean(img[:, :, 1])
        mean[2] += np.mean(img[:, :, 2])
        std[0] += np.std(img[:, :, 0])
        std[1] += np.std(img[:, :, 1])
        std[2] += np.std(img[:, :, 2])
        print(step)
    mean /= len(os.listdir(data_folder))
    std /= len(os.listdir(data_folder))
    np.savetxt('street_mean_std.txt', (mean, std), fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    # create_dataset()
    cal_mean_and_std()
