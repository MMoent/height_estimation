import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import *
from datetime import datetime
from Model import UNet
from SatStreetDataset import SatStreetDataset
from geo_transform import get_pano_depth
from equilib import equi2pers
from loss import ssim

from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR



def scale_invariant_loss(output, target):
    # di = output - target
    di = target - output
    n = (256 * 256)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    return loss.mean()


def main():
    epochs = 100
    bs = 1
    lr = 0.01
    weight_decay = 1e-4

    height, width = 256, 256
    fov = 90.0

    train_dataset = SatStreetDataset(root='./Data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)

    val_dataset = SatStreetDataset(root='./Data', train=False, test=False)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=8)

    # model = UNet(3, 1)

    model = smp.Unet(
        encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    model = model.cuda()
    # checkpoint = 'Experiments/20220607_142919/model_best.pth'
    # model.load_state_dict(torch.load(checkpoint))

    criterion_1 = L1Loss().cuda()
    criterion_2 = MSELoss().cuda()

    optimizer = Adam(model.parameters(),
                     lr=lr,
                     weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.5)

    best_rmse = 1e9
    for ep in range(1, epochs+1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        overall_loss = []
        for step, (sat_im, sat_mask, street_mask) in loop:
            sat_im, sat_mask, street_mask = sat_im.cuda(), sat_mask.unsqueeze(1).cuda(), street_mask.cuda()
            logits = model(sat_im)
            loss_sat = scale_invariant_loss(logits, sat_mask)

            # satellite to street-view
            t_logits = get_pano_depth(logits)
            t_logits.retain_grad()
            loss_pers = []
            loss_ssim = []
            for h in range(4):
                rots = [{'roll': 0., 'pitch': 5./180*np.pi, 'yaw': h * 90 / 180 * np.pi}] * bs
                pers = equi2pers(t_logits, rots, height, width, fov, z_down=True)
                pers_midas = street_mask[:, h, ...]

                if torch.equal(pers_midas, torch.zeros_like(pers_midas)):
                    continue
                loss_pers.append(scale_invariant_loss(pers, pers_midas))
                loss_ssim.append(ssim(pers, pers_midas, 128.0))

            total_loss = 0.5 * loss_sat
            if len(loss_pers) != 0:
                coef = 1 / len(loss_pers)
                for l, s in zip(loss_pers, loss_ssim):
                    total_loss += coef * (l + s)

            overall_loss.append(total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{ep}/{epochs}]')
            loop.set_postfix(loss=total_loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            total_rmse = []
            for sat_im, sat_mask in val_loader:
                sat_im, sat_mask = sat_im.cuda(), sat_mask.unsqueeze(1).cuda()
                logits = model(sat_im)
                rmse = torch.sqrt(criterion_2(logits, sat_mask))
                total_rmse.append(rmse)
            averaged_loss = sum(overall_loss) / len(overall_loss)
            averaged_rmse = sum(total_rmse) / len(total_rmse)
            print(
                f'Epoch {ep} -- Loss: {averaged_loss:.4f} RMSE: {averaged_rmse.item():.4f} Best RMSE: {best_rmse:.4f}')

            if averaged_rmse < best_rmse:
                best_rmse = averaged_rmse
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']

                save_dir = os.path.join('Experiments_with_streetview', datetime.now().strftime('%Y%m%d_%H%M%S'))
                os.mkdir(save_dir)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, 'model_best.pth'))

                indeces = f'epoch:{ep} batch_size:{bs} lr:{current_lr} rmse:{averaged_rmse:.4f} loss:{averaged_loss:.4f}'
                with open(os.path.join(save_dir, 'indeces.txt'), 'w') as f:
                    f.write(indeces)

                print(indeces)
                print('Model saved.\n')
        lr_scheduler.step()


if __name__ == "__main__":
    main()
