import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import *
from Model import UNet
from SatStreetDataset import SatStreetDataset

from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss


def test(save_dir, encoder_name):
    bs = 16

    test_dataset = SatStreetDataset(root='./Data', train=False, test=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    criterion1 = L1Loss().cuda()
    criterion2 = MSELoss().cuda()

    model = smp.Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    ).cuda()

    checkpoint = os.path.join(save_dir, 'model_best.pth')
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint).items()})

    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    total_mae, total_rmse = [], []
    with torch.no_grad():
        model.eval()
        for step, (image, mask) in loop:
            image, mask = image.cuda(), mask.cuda().unsqueeze(1)
            prediction = model(image)
            prediction = torch.where(prediction < 0, 0, prediction)
            mae = criterion1(prediction, mask)
            rmse = torch.sqrt(criterion2(prediction, mask))
            total_mae.append(mae)
            total_rmse.append(rmse)
        averaged_rmse = sum(total_rmse) / len(total_rmse)
        averaged_mae = sum(total_mae) / len(total_mae)

        indeces = f'Testing batch size:{bs} RMSE: {averaged_rmse:.4f} MAE: {averaged_mae:.4f}'
        with open(os.path.join(save_dir, 'test_result.txt'), 'w') as f:
            f.write(indeces)
        print(f'Averaged RMSE: {averaged_rmse:.4f}')
        print(f'Averaged MAE: {averaged_mae:.4f}')


if __name__ == "__main__":
    test('Experiments/efficientnet-b7_NO_STREET_SIE', 'efficientnet-b7')
