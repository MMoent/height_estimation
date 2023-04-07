import os
import glob
import cv2
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def main():
    model_type = "DPT_Large"  # MiDaS v3 - Large  (highest accuracy, slowest inference speed)

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

        for step, i in enumerate(glob.glob('street_sat/'+'*.png')):
            s_id = os.path.join('street_view_images', os.path.split(i)[-1][:-4])
            for h in range(4):
                street_path = s_id + '_' + str(h * 90) + '.jpeg'
                if not os.path.exists(street_path):
                    continue
                img = cv2.cvtColor(cv2.imread(street_path), cv2.COLOR_BGR2RGB)
                plt.subplot(1,2,1)
                plt.imshow(img)

                img = img[:490, :]
                # img = cv2.resize(img, (256, 256))
                img = transform(img).to(device)
                prediction = model(img)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(256, 256),
                    mode="bicubic",
                    align_corners=True,
                ).squeeze()
                output = prediction.cpu().numpy()

                b = output < 0.1
                output[b] = 0

                out_path = os.path.join('street_view_depth_watermark_removed', os.path.split(street_path)[-1][:-5]+'.tif')
                plt.subplot(1, 2, 2)
                plt.imshow(output)
                plt.show()
                print(out_path, 'done')
                # cv2.imwrite(out_path, output)


def degrade_depth(path):
    im = cv2.imread(path, -1)

    b = im < 0.1
    im[b] = 0

    height, width = 256, 256
    plt.subplot(2, 2, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.title('original')

    k = np.ones((3, 3))
    im_filtered = cv2.medianBlur(im, 5)

    plt.subplot(2, 2, 2)
    plt.imshow(im_filtered)
    plt.axis('off')
    plt.title('filtered')

    degraded_im, degraded_im_filtered = im, im_filtered
    for j in range(width):
        st = 0
        s1, s2 = degraded_im[st, j], degraded_im_filtered[st, j]
        for i in range(st+1, height):
            if degraded_im[i, j] < s1:
                degraded_im[i, j] = s1
            else:
                s1 = degraded_im[i, j]

            if degraded_im_filtered[i, j] < s2:
                degraded_im_filtered[i, j] = s2
            else:
                s2 = degraded_im_filtered[i, j]

    plt.subplot(2, 2, 3)
    plt.imshow(degraded_im)
    plt.axis('off')
    plt.title('original degraded')

    plt.subplot(2, 2, 4)
    plt.imshow(degraded_im_filtered)
    plt.axis('off')
    plt.title('filtered degraded')

    cv2.imwrite(os.path.join('./street_view_depth_degraded', os.path.split(path)[-1]), degraded_im_filtered)
    plt.savefig(os.path.join('./street_view_depth_degraded', os.path.split(path)[-1][:-4]+'.png'), dpi=300)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    # main()
    path = './street_view_images'
    for fn in os.listdir(path):
        im_path = os.path.join(path, fn)

