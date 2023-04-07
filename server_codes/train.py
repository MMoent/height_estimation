import os
from datetime import datetime
import random
import numpy as np
import torch
from test import test
import segmentation_models_pytorch as smp

from equilib import equi2pers
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import *

from loss import ssim
from Model import UNet
from SatStreetDataset import SatStreetDataset
from geo_transform import get_pano_depth


def scale_invariant_loss(output, target):
    # di = output - target
    di = target - output
    n = (256 * 256)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    return loss.mean()


def train(sat_weight, st_weight, w1, w2, loss_func):
    epochs = 1000
    bs = 32
    lr = 0.01
    weight_decay = 1e-4

    height, width = 256, 256
    fov = 90.0

    train_dataset = SatStreetDataset(root='./Data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)

    val_dataset = SatStreetDataset(root='./Data', train=False, test=False)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=8)

    # model = UNet(3, 1)
    encoder_name = "efficientnet-b7"
    model = smp.Unet(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    criterion = None
    if loss_func == 'L1':
        criterion = L1Loss().cuda()
    elif loss_func == 'L2':
        criterion = MSELoss().cuda()
    elif loss_func == 'SIE':
        criterion = scale_invariant_loss

    MSE = MSELoss().cuda()

    optimizer = Adam(model.parameters(),
                     lr=lr,
                     weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.5)

    if st_weight:
        save_dir_path = os.path.join('Experiments', f'{encoder_name}_{sat_weight}SAT_{st_weight}ST_{w1}{loss_func}_{w2}SSIM')
    else:
        save_dir_path = os.path.join('Experiments', f'{encoder_name}_NO_STREET_{loss_func}')

    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    print(f'Saving model in {save_dir_path}...')
    best_rmse = 1e9
    log = []
    for ep in range(1, epochs + 1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        overall_loss = []
        for step, (sat_im, sat_mask, street_mask) in loop:
            sat_im, sat_mask, street_mask = sat_im.cuda(), sat_mask.unsqueeze(1).cuda(), street_mask.cuda()
            logits = model(sat_im)

            loss_sat = criterion(logits, sat_mask)
            total_loss = sat_weight * loss_sat

            if st_weight != 0:
                # satellite to street-view
                t_logits = get_pano_depth(logits, (height, 2 * width))

                loss_pers = []
                loss_pers_ssmi = []

                for h in range(4):
                    rots = [{'roll': 0., 'pitch': 10. / 180 * np.pi, 'yaw': h * 90 / 180 * np.pi}] * bs
                    pers = equi2pers(t_logits, rots, height, width, fov, z_down=True)
                    pers_midas = street_mask[:, h, ...]
                    if torch.equal(pers_midas, torch.zeros_like(pers_midas)):
                        continue
                    # loss between transformed street-view depth
                    loss_pers.append(criterion(pers, pers_midas))
                    loss_pers_ssmi.append(torch.clamp((1 - ssim(pers, pers_midas, val_range=128.)) * 0.5, 0, 1))

                if len(loss_pers) != 0:
                    coef = st_weight / len(loss_pers)
                    for lt, st in zip(loss_pers, loss_pers_ssmi):
                        total_loss += coef * (w1 * lt + w2 * st)

            overall_loss.append(total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{ep}/{epochs}]')
            loop.set_postfix(loss=total_loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            total_val_loss, total_rmse = [], []
            for sat_im, sat_mask in val_loader:
                sat_im, sat_mask = sat_im.cuda(), sat_mask.unsqueeze(1).cuda()
                logits = model(sat_im)

                total_val_loss.append(criterion(logits, sat_mask))
                total_rmse.append(torch.sqrt(MSE(logits, sat_mask)))

            averaged_train_loss = sum(overall_loss) / len(overall_loss)
            averaged_train_loss = averaged_train_loss.item()

            averaged_val_loss = sum(total_val_loss) / len(total_val_loss)
            averaged_val_loss = averaged_val_loss.item()

            averaged_rmse = sum(total_rmse) / len(total_rmse)
            averaged_rmse = averaged_rmse.item()

            log.append([averaged_train_loss, averaged_val_loss, averaged_rmse])

            print(
                f'Epoch {ep} -- Train loss: {averaged_train_loss:.4f} Val loss: {averaged_val_loss:.4f}',
                f'Val RMSE: {averaged_rmse:.4f}',
                f'Best RMSE: {best_rmse:.4f}'
            )

            if averaged_rmse < best_rmse:
                best_rmse = averaged_rmse
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                torch.save(model.state_dict(),
                           os.path.join(save_dir_path, 'model_best.pth'))

                indeces = f'epoch:{ep} batch_size:{bs} lr:{current_lr} ' \
                          f'RMSE: {averaged_rmse:.4f} Val Loss: {averaged_val_loss:.4f}'
                with open(os.path.join(save_dir_path, 'indeces.txt'), 'w') as f:
                    f.write(indeces)

                print(indeces)
                print(f'Sat_weight={sat_weight}, st_weight={st_weight}. Model saved to {save_dir_path}.\n')
        lr_scheduler.step()

    np.save(os.path.join(save_dir_path, 'log.npy'), np.array(log))
    print('testing...')
    test(save_dir_path, encoder_name)
    print('testing over')
    print('--------------------------------------------------------------------')


if __name__ == "__main__":
    SEED_NUM = 0
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)

    loss_func = 'SIE'

    # w1, w2 = 0.1, 1.0
    # sat_weight = [1.0, 0.6, 0.2]
    # st_weight = [1.0, 2.0, 3.0]
    # for sat in sat_weight:
    #     for st in st_weight:
    #         train(sat, st, w1, w2, loss_func)

    sat_weight, st_weight = 1.0, 0.0
    # w1 = [0.2, 0.6, 1.0]
    # w2 = [1.0, 0.6, 0.2]

    train(sat_weight, st_weight, 0.6, 0.6, loss_func)

    # sat_weight, st_weight = 1.0, 1.0
    # train(sat_weight, st_weight, w1, w2, loss_func)
