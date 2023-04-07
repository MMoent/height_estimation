import os
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from Data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Model import UNet
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import *


def scale_invariant_loss(output, target):
    # di = output - target
    di = target - output
    n = (256 * 256)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    return loss.mean()


def train(model, device, train_dataloader, val_dataloader, epochs, batch_size, lr, weight_decay, save_dir):
    model = model.to(device)
    criterion_1 = L1Loss().to(device)
    criterion_2 = MSELoss().to(device)
    optimizer = Adam(model.parameters(),
                     lr=lr,
                     weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.5)

    best_rmse = 9999
    for epoch in range(1, epochs + 1):
        model.train()
        overall_loss = []
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, (image, label) in loop:
            image = image.to(device)
            label = label.unsqueeze(1).to(device)
            logits = model(image)
            # loss = criterion_1(logits, label)
            loss = scale_invariant_loss(logits, label).to(device)
            overall_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            total_rmse = []
            for image, label in val_dataloader:
                image, label = image.to(device), label.unsqueeze(1).to(device)
                logits = model(image)
                rmse = torch.sqrt(criterion_2(logits, label))
                total_rmse.append(rmse)

            averaged_loss = sum(overall_loss) / len(overall_loss)
            averaged_rmse = sum(total_rmse) / len(total_rmse)
            print(
                f'Epoch {epoch} -- Loss: {averaged_loss:.4f} RMSE: {averaged_rmse.item():.4f} Best RMSE: {best_rmse:.4f}')

            if averaged_rmse < best_rmse:
                best_rmse = averaged_rmse
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']

                torch.save(model.state_dict(),
                           os.path.join(save_dir, 'model_best.pth'))

                indeces = f'epoch:{epoch} batch_size:{batch_size} lr:{current_lr} rmse:{averaged_rmse:.4f} loss:{averaged_loss:.4f}'
                with open(os.path.join(save_dir, 'indeces.txt'), 'w') as f:
                    f.write(indeces)

                print(indeces)
                print('Model saved.\n')

            lr_scheduler.step()


def test(model, device, test_dataloader, checkpoint):
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    criterion = MSELoss().to(device)
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    total_rmse = []
    with torch.no_grad():
        for step, (image, label) in loop:
            image, label = image.to(device), label.unsqueeze(1).to(device)
            prediction = model(image)
            rmse = torch.sqrt(criterion(prediction, label))
            total_rmse.append(rmse)
        averaged_rmse = sum(total_rmse) / len(total_rmse)
        print(f'Averaged RMSE: {averaged_rmse:.4f}')


def predict(model, device, checkpoint, input):
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    with torch.no_grad():
        output = model(input.unsqueeze(0).to(device))
        ret = output.squeeze().to('cpu').numpy()
    return ret


def main():
    stage = 'test'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    batch_size = 8
    lr = 0.01
    weight_decay = 1e-4

    root = './Data'
    pred_save_dir = './Data/Pred_SIE_cropped'
    checkpoint = 'Experiments/20220520_211939/model_best.pth'
    model = UNet(3, 1)

    if stage == 'train':
        save_dir = os.path.join('Experiments', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.mkdir(save_dir)
        train_dataset = Dataset(root=root, train=True, test=False)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_dataset = Dataset(root=root, train=False, test=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        train(model=model, device=device, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
              epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, save_dir=save_dir)
    elif stage == 'test':
        test_dataset = Dataset(root=root, train=False, test=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=4)
        test(model, device, test_dataloader, checkpoint)
    elif stage == 'predict':
        img_ids = [i[:-8] for i in glob.glob(os.path.join(root, 'Test_cropped', '*.png'))]
        for img_id in img_ids:
            img = cv2.imread(img_id + '_rgb.png')
            gt = cv2.imread(img_id + '_ndsm.tif', -1)
            transform = A.Compose([
                A.Normalize(mean=(0.3947, 0.4172, 0.4097), std=(0.1630, 0.1413, 0.1224)),
                ToTensorV2()
            ])
            transformed_img = transform(image=img)
            input = transformed_img['image']
            ret = predict(model=model, device=device, checkpoint=checkpoint, input=input)
            out_path = os.path.join(pred_save_dir, os.path.split(img_id)[1] + '_ndsm_pred.tif')
            cv2.imwrite(out_path, ret)
            print(out_path, 'done')
            # plt.subplot(1, 3, 1)
            # plt.imshow(img)
            # plt.subplot(1, 3, 2)
            # plt.imshow(gt)
            # plt.subplot(1, 3, 3)
            # plt.imshow(ret)
            # plt.show()
        # predict(data_dir=root)


if __name__ == '__main__':
    main()
    
