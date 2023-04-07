import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from test_with_st import test
from torch.nn import MSELoss, L1Loss, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from ResUnet import ResUnet
from tqdm import *
from SatStreetDataset import SatStreetDataset


def scale_invariant_loss(output, target):
    # di = output - target
    di = target - output
    n = (256 * 256)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    return loss.mean()


def train(loss_func):
    epochs = 10
    bs = 8
    lr = 0.01
    weight_decay = 1e-4

    train_dataset = SatStreetDataset(root='./Data', train=True, test=False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)

    val_dataset = SatStreetDataset(root='./Data', train=False, test=False)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=8)

    model = ResUnet(inc1=3, inc2=12, outc=1)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    criterion = None
    if loss_func == 'l1':
        criterion = L1Loss().cuda()
    elif loss_func == 'l2':
        criterion = MSELoss().cuda()
    elif loss_func == 'sie':
        criterion = scale_invariant_loss
    MSE = MSELoss().cuda()

    optimizer = Adam(model.parameters(),
                     lr=lr,
                     weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.5)

    save_dir_path = os.path.join('Experiments', f'COMBINED_{loss_func.upper()}')
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    print(f'Saving model in {save_dir_path}...')

    best_rmse = 1e9
    log = []
    for ep in range(1, epochs + 1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        train_loss = []
        for step, (sat_im, sat_mask, street_ims) in loop:
            sat_im, sat_mask, street_ims = sat_im.cuda(), sat_mask.unsqueeze(1).cuda(), street_ims.cuda()
            logits = model(sat_im, street_ims)
            loss = criterion(logits, sat_mask)
            train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{ep}/{epochs}]')
            loop.set_postfix(loss=loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            val_loss, val_rmse = [], []
            for sat_im, sat_mask, street_ims in val_loader:
                sat_im, sat_mask, street_ims = sat_im.cuda(), sat_mask.unsqueeze(1).cuda(), street_ims.cuda()
                logits = model(sat_im, street_ims)
                val_loss.append(criterion(logits, sat_mask))
                val_rmse.append(torch.sqrt(MSE(logits, sat_mask)))
            # log training & validation
            ave_train_loss = sum(train_loss) / len(train_loss)
            ave_val_loss = sum(val_loss) / len(val_loss)
            ave_val_rmse = sum(val_rmse) / len(val_rmse)
            log.append([ave_train_loss.item(), ave_val_loss.item(), ave_val_rmse.item()])
            print(
                f'Epoch {ep} -- Train loss: {ave_train_loss:.4f} Val loss: {ave_val_loss:.4f}',
                f'Val RMSE: {ave_val_rmse:.4f}',
                f'Best RMSE: {best_rmse:.4f}'
            )
            if ave_val_rmse < best_rmse:
                best_rmse = ave_val_rmse
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                torch.save(model.state_dict(),
                           os.path.join(save_dir_path, 'model_best.pth'))
                indeces = f'epoch:{ep} batch_size:{bs} lr:{current_lr} ' \
                          f'RMSE: {ave_val_rmse:.4f} Val Loss: {ave_val_loss:.4f}'
                with open(os.path.join(save_dir_path, 'indeces.txt'), 'w') as f:
                    f.write(indeces)
                print(indeces,
                      f'Model saved to {save_dir_path}.\n')
        lr_scheduler.step()

    np.save(os.path.join(save_dir_path, 'log.npy'), np.array(log))
    print('testing...')
    test(save_dir_path)
    print('testing over')
    print('--------------------------------------------------------------------')


if __name__ == "__main__":
    SEED_NUM = 0
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)

    train(loss_func='sie')
