import os
import matplotlib
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from Model import UNet, UNet3D
from albumentations.pytorch import ToTensorV2
from torch.nn import MSELoss, L1Loss
from geo_transform import get_pano_depth
from equilib import equi2pers

# matplotlib.use('Agg')


def main(checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = A.Compose([
        A.Normalize(mean=(0.3947, 0.4172, 0.4097), std=(0.1630, 0.1413, 0.1224)),
        ToTensorV2()
    ])

    checkpoint = os.path.join(checkpoint_dir, 'model_best.pth')
    model = UNet(3, 1).to(device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint).items()})
    model.eval()

    criterion = MSELoss().to(device)

    im_ids = []
    with open('Data/test.txt', 'r') as f:
        for im_id in f:
            im_ids.append(im_id[:-1])

    save_dir = os.path.join(checkpoint_dir, 'test_output')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for e, im_id in enumerate(im_ids):
            im_path = os.path.join('./Data/street_sat', im_id + '.png')
            gt_path = os.path.join('./Data/street_sat', im_id + '.tif')
            im = cv2.imread(im_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(gt_path, -1)

            t = transforms(image=im, mask=mask)
            model_input = t['image'].unsqueeze(0).to(device)
            model_mask = t['mask'].view(1, 1, 256, 256).to(device)
            logits = model(model_input)
            logits = torch.where(logits < 0, 0, logits)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.voxels(logits_3d)
            # plt.show()

            t_logits = get_pano_depth(logits)
            t_mask = get_pano_depth(model_mask)

            t_logits = t_logits.squeeze().cpu().numpy()
            t_mask = t_mask.squeeze().cpu().numpy()

            for h in range(4):
                rots = [{'roll': 0., 'pitch': 5/180*np.pi, 'yaw': h * 90 / 180 * np.pi}]
                pers = equi2pers(t_logits, rots, 256, 256, 90, z_down=True)
                pers_mask = equi2pers(t_mask, rots, 256, 256, 90, z_down=True)
                pers = pers.squeeze().cpu().numpy()
                pers_mask = pers_mask.squeeze().cpu().numpy()

                plt.subplot(2, 4, h+1)
                plt.imshow(pers)
                plt.axis('off')
                plt.subplot(2, 4, h+5)
                plt.imshow(pers_mask)
                plt.axis('off')

            # plt.savefig(os.path.join(save_dir, im_id+'.png'), dpi=300)
            # plt.close('all')
            print(e, '/', len(im_ids))
            output = logits.squeeze().cpu().numpy()
            # plt.subplot(1,2,1)
            # plt.imshow(mask)
            # plt.subplot(1,2,2)
            # plt.imshow(output)
            # plt.show()
            # plt.close("all")
            cv2.imwrite(os.path.join(save_dir, im_id+'.tif'), output)


if __name__ == "__main__":
    checkpoint_dir = './Experiments/1.0SAT_2.0ST_1.0SIE_0SSIM'
    main(checkpoint_dir)
