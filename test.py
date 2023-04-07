import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from datetime import datetime
from Model import UNet
from SatStreetDataset import SatStreetDataset

from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # device_ids = [0, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = 4

    test_dataset = SatStreetDataset(root='./Data', train=False, test=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    criterion = MSELoss().to(device)

    model = UNet(3, 1).to(device)
    checkpoint = './Experiments/no_street/model_best.pth'
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint).items()})
    # model.load_state_dict(torch.load(checkpoint))
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    total_rmse = []
    with torch.no_grad():
        for step, (image, mask) in loop:
            image, mask = image.to(device), mask.to(device).unsqueeze(1)
            prediction = model(image)
            prediction = torch.where(prediction < 0, 0, prediction)
            rmse = torch.sqrt(criterion(prediction, mask))
            total_rmse.append(rmse)
        averaged_rmse = sum(total_rmse) / len(total_rmse)
        print(f'Averaged RMSE: {averaged_rmse:.4f}')


if __name__ == "__main__":
    main()
