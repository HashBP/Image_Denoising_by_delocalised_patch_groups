import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import os
from env import *
from compare_images_csv import *
import os
import shutil


def start_filter():
    for i in range(len(initial_delete_folder)):
        for filename in os.listdir(initial_delete_folder[i]):
            file_path = os.path.join(initial_delete_folder[i], filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    for i in range(len(NOISE)):
        os.makedirs(f"Filtered_channels/Filtered_{NOISE[i]}")
        for j in range(len(NOISE_MAGNITUDE)):
            path = f"noisy_images/{NOISE[i]}{int(NOISE_MAGNITUDE[j]*100)}.jpg"
            noisy_img = cv2.imread(path)
            noisy_img = noisy_img.astype(np.float32)
            sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))
            denoise_img = denoise_nl_means(
                noisy_img, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=2, multichannel=True)
            cv2.imwrite(
                f"Filtered_channels/Filtered_{NOISE[i]}/{NOISE_MAGNITUDE[j]}.jpg", denoise_img)
            print(
                f"Done Filtering of {NOISE[i]} {int(NOISE_MAGNITUDE[j]*100)}%......")
            compare_images(
                f"Filtered_channels/Filtered_{NOISE[i]}/{NOISE_MAGNITUDE[j]}.jpg", int(NOISE_MAGNITUDE[j]*100))
        make_csv(f"{NOISE[i]}",img[k])
        percent_n.clear()
        mse_calc.clear()
        uqi_calc.clear()
        psnr_calc.clear()
        ssim_calc.clear()
        scc_calc.clear()
        rase_calc.clear()

start_filter()