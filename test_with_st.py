import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import *
from ResUnet import ResUnet
from SatStreetDataset import SatStreetDataset

from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss


def test(save_dir):
    bs = 8

    test_dataset = SatStreetDataset(root='./Data', train=False, test=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    criterion1 = L1Loss().cuda()
    criterion2 = MSELoss().cuda()

    model = ResUnet(inc1=3, inc2=12, outc=1).cuda()

    checkpoint = os.path.join(save_dir, 'model_best.pth')
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint).items()})

    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    total_mae, total_rmse = [], []
    with torch.no_grad():
        model.eval()
        for step, (image, mask) in loop:
            sat_im, sat_mask, street_ims = sat_im.cuda(), sat_mask.unsqueeze(1).cuda(), street_ims.cuda()
            prediction = model(sat_im, street_ims)
            prediction = torch.where(prediction < 0, 0, prediction)
            mae = criterion1(prediction, mask)
            rmse = torch.sqrt(criterion2(prediction, mask))
            total_mae.append(mae)
            total_rmse.append(rmse)
        ave_rmse = sum(total_rmse) / len(total_rmse)
        ave_mae = sum(total_mae) / len(total_mae)

        indeces = f'Testing batch size:{bs} RMSE: {ave_rmse:.4f} MAE: {ave_rmse:.4f}'
        with open(os.path.join(save_dir, 'test_result.txt'), 'w') as f:
            f.write(indeces)
        print(f'Averaged RMSE: {ave_rmse:.4f}')
        print(f'Averaged MAE: {ave_rmse:.4f}')


if __name__ == "__main__":
    pass
