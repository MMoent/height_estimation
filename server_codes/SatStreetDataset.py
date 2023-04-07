import os
import glob
import numpy as np
import cv2
import albumentations as A
from ast import literal_eval

import torch
from albumentations.pytorch import ToTensorV2
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class SatStreetDatasetOld(data.Dataset):
    def __init__(self, root, train=True, test=False, transforms=None, train_with_street=True):
        self.root = root
        self.train = train
        self.train_with_street = train_with_street
        self.test = test
        # load image file name
        self.imgs = []
        target_file = 'train' if train else ('test' if test else 'val')
        with open(os.path.join(root, target_file+'.txt'), 'r') as f:
            for line in f:
                self.imgs.append(line[:-1])
        # street view depth
        self.street = os.listdir(os.path.join(root, 'street_view_depth'))

        if not transforms:
            sat_normalize = A.Normalize(mean=(0.3947, 0.4172, 0.4097), std=(0.1630, 0.1413, 0.1224))
            street_normalize = A.Normalize(mean=(0.4777, 0.4884, 0.4794), std=(0.2728, 0.2716, 0.2918))
            self.street_transforms = A.Compose([
                A.Resize(256, 256),
                # street_normalize,
                ToTensorV2()
            ])
            if not self.train:
                self.transforms = A.Compose([
                    sat_normalize,
                    ToTensorV2(),
                ])
            else:
                self.transforms = A.Compose([
                    # A.Resize(256, 256),
                    # A.RandomResizedCrop(256, 256, p=0.5),
                    # A.HorizontalFlip(p=0.5),
                    sat_normalize,
                    ToTensorV2(),
                ])

    def __getitem__(self, item):
        sat_im = cv2.imread(os.path.join(self.root, 'street_sat', self.imgs[item]+'.png'))
        sat_im = cv2.cvtColor(sat_im, cv2.COLOR_BGR2RGB)
        sat_mask = cv2.imread(os.path.join(self.root, 'street_sat', self.imgs[item]+'.tif'), -1)
        t = self.transforms(image=sat_im, mask=sat_mask)

        if self.train and self.train_with_street:
            street_mask = torch.zeros((4, 1, 256, 256)).float()
            for h in range(4):
                name = self.imgs[item] + '_' + str(h*90) + '.tif'
                if name in self.street:
                    st_depth = cv2.imread(os.path.join(self.root, 'street_view_depth', name), -1)
                    b = st_depth > 0
                    st_depth[b] -= st_depth[b].min()
                    st_depth[b] /= (st_depth[b].max() - st_depth[b].min())
                    transformed_im = self.street_transforms(image=st_depth)['image']
                    street_mask[h, ...] = transformed_im
            return t['image'], t['mask'], street_mask
        else:
            return t['image'], t['mask']

    def __len__(self):
        return len(self.imgs)


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
    train_dataset = SatStreetDataset(root='./Data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for idx, (image, mask, st_images) in enumerate(train_loader):
        pass
