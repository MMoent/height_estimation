import os
import glob
import numpy as np
import albumentations as A
import cv2
import torch
import matplotlib.pyplot as plt
from Model import UNet
from albumentations.pytorch import ToTensorV2

def main():
    data_dir = './sample'

    model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # model = UNet(3, 1).to(device)
    # checkpoint = './20221029_025242/model_best.pth'
    # model.load_state_dict(torch.load(checkpoint))
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    with torch.no_grad():
        for sub_dir in os.listdir(data_dir):
            names = [i for i in glob.glob(os.path.join(data_dir, sub_dir, sub_dir+'*')) if i[-6] == '0']
            for name in names:
                img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, (256, 256))
                plt.subplot(1, 2, 1)
                plt.imshow(img)

                img = transform(img).to(device)
                prediction = model(img)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(256, 256),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                output = prediction.cpu().numpy()
                plt.subplot(1, 2, 2)
                plt.imshow(output)
                plt.savefig(name[:-5]+'_midas.jpeg', dpi=300)
                plt.show()
                # c = input()


if __name__ == "__main__":
    main()
