import os
import glob
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, root, train=True, test=False, transforms=None):
        self.train = train
        self.test = test
        self.imgs = []
        target_dir = 'Train' if train else ('Test' if test else 'Val')
        self.imgs = glob.glob(os.path.join(root, target_dir, '*.png'))
        if not transforms:
            normalize = A.Normalize(mean=(0.3947, 0.4172, 0.4097),
                                    std=(0.1630, 0.1413, 0.1224))
            if not self.train:
                self.transforms = A.Compose([
                    normalize,
                    ToTensorV2(),
                ])
            else:
                self.transforms = A.Compose([
                    A.Resize(256, 256),
                    A.RandomResizedCrop(256, 256, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    normalize,
                    ToTensorV2(),
                ])

    def __getitem__(self, item):
        image = cv2.imread(self.imgs[item])
        mask = cv2.imread(self.imgs[item][:-8]+'_ndsm.tif', -1)
        transformed = self.transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        return transformed_image, transformed_mask

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train_dataset = Dataset(root='./', train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for idx, (image, label) in enumerate(train_loader):
        print(image.shape, label.shape)
        show_label = label[0, ...].numpy()
        show_im = image[0, ...].permute(1, 2, 0).numpy()
        plt.figure()
        plt.imshow(show_label)
        plt.figure()
        plt.imshow(show_im)
        plt.show()
        c = input()
