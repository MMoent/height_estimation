import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from ast import literal_eval
from torch.optim import Adam

import torch
from albumentations.pytorch import ToTensorV2
from torch.utils import data
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss, BCELoss
from ResUnet import ResUnet


class SatStreetDataset(data.Dataset):
    def __init__(self, root, train=True, test=False):
        self.root = root

        # load image file name
        self.imgs = []
        target_file = 'train' if train else ('test' if test else 'val')
        with open(os.path.join(root, target_file+'.txt'), 'r') as f:
            for line in f:
                self.imgs.append(line.strip())

        # street view image
        self.street = os.listdir(os.path.join(root, 'street_view_images'))

        sat_normalize = A.Normalize(mean=(0.3947, 0.4172, 0.4097), std=(0.1630, 0.1413, 0.1224))
        street_normalize = A.Normalize(mean=(0.5094, 0.5206, 0.5120), std=(0.2818, 0.2814, 0.3035))
        self.street_transforms = A.Compose([
            A.Resize(256, 256),
            street_normalize,
            ToTensorV2()
        ])
        self.transforms = A.Compose([
            sat_normalize,
            ToTensorV2(),
        ])

    def __getitem__(self, item):
        sat_im = cv2.imread(os.path.join(self.root, 'street_sat', self.imgs[item]+'.png'))
        sat_im = cv2.cvtColor(sat_im, cv2.COLOR_BGR2RGB)
        sat_mask = cv2.imread(os.path.join(self.root, 'street_sat', self.imgs[item]+'.tif'), -1)
        t = self.transforms(image=sat_im, mask=sat_mask)

        st_imgs = torch.zeros((12, 256, 256)).float()
        for h in range(4):
            fn = self.imgs[item] + '_' + str(h*90) + '.jpeg'
            if fn in self.street:
                st_im = cv2.imread(os.path.join(self.root, 'street_view_images', fn))
                st_im = cv2.cvtColor(st_im, cv2.COLOR_BGR2RGB)
                transformed_im = self.street_transforms(image=st_im)['image']
                st_imgs[h*3:h*3+3, ...] = transformed_im

        return t['image'], t['mask'], st_imgs

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # x = torch.randn((2, 3, 256, 256))
    model = ResUnet(3, 12, 1)
    # y = model(x)
    criterion = L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_dataset = SatStreetDataset(root='./Data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for ep in range(100):
        model.train()
        for idx, (sat_im, sat_mask, st_ims) in enumerate(train_loader):
            sat_mask = sat_mask.unsqueeze(1)
            # print(sat_im.shape, sat_mask.shape, st_ims.shape)
            logits = model(sat_im, st_ims)
            plt.imshow(logits[0, ...].squeeze().detach().numpy())
            plt.show()
            plt.close('all')
            loss = criterion(logits, sat_mask)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
