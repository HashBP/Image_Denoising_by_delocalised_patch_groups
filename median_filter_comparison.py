import os
import pandas as pd
from env import *
from tabulate import tabulate
from PIL import Image, ImageFilter
import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from sewar.full_ref import mse, psnr, ssim, uqi, scc, rase

# folder = ["coin", "leaves", "leena", "sky", "train"]
folder = ["coin", "leaves", "leena", "sky", "train"]
noise = ["gaussian_noise", "poisson_noise", "speckle_noise", "sp_noise"]
# noise = ["gaussian_noise","leaves"]

filter = []
mse_calc = []
psnr_calc = []
uqi_calc = []
scc_calc = []
rase_calc = []
ssim_calc = []
noise_name=[]
pic_name=[]

for i in range(len(folder)):
    for j in range(len(noise)):

        PATH = f"noisy_images/leena/{noise[j]}80.jpg"

        # My Filter
        noisy_img = cv2.imread(PATH)
        noisy_img = noisy_img.astype(np.float32)
        sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))
        denoise_img = denoise_nl_means(
            noisy_img, h=6.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=2, multichannel=True)
        cv2.imwrite("My_Filter.jpg", denoise_img)

        # Median Filter
        im1 = Image.open(PATH)
        im2 = im1.filter(ImageFilter.MedianFilter(size=5))
        im2.save("Median_Filter.jpg")

        # ///////////////////////////

        original_img = f"test_images/lenna.jpg"
        org_img = cv2.imread(original_img)
        comp_img1 = cv2.imread("My_Filter.jpg")
        comp_img2 = cv2.imread("Median_Filter.jpg")

        m = mse(org_img, comp_img1)
        u = uqi(org_img, comp_img1)
        s = scc(org_img, comp_img1)
        r = rase(org_img, comp_img1)
        p = psnr(org_img, comp_img1)
        ss = ssim(org_img, comp_img1)[0]

        filter.append("My Filter")
        pic_name.append(folder[i])
        mse_calc.append(m)
        uqi_calc.append(u)
        psnr_calc.append(p)
        ssim_calc.append(ss)
        scc_calc.append(s)
        rase_calc.append(r)
        noise_name.append(noise[j])

        m = mse(org_img, comp_img2)
        u = uqi(org_img, comp_img2)
        s = scc(org_img, comp_img2)
        r = rase(org_img, comp_img2)
        p = psnr(org_img, comp_img2)
        ss = ssim(org_img, comp_img2)[0]

        filter.append("Median Filter")
        pic_name.append(folder[i])
        mse_calc.append(m)
        uqi_calc.append(u)
        psnr_calc.append(p)
        ssim_calc.append(ss)
        scc_calc.append(s)
        rase_calc.append(r)
        noise_name.append(noise[j])
    dict = {
        "Picture":pic_name,
        "Noise":noise_name,
        "Filter": filter,
        "MSE⬇️": mse_calc,
        "RASE⬇️": rase_calc,
        "UQI⬆️": uqi_calc,
        "PSNR⬆️": psnr_calc,
        "SSIM⬆️": ssim_calc,
        "SCC⬆️": scc_calc
    }
    df = pd.DataFrame(dict)
    df.to_csv(
        f"{folder[i]}_80_folder_data.csv", index=False)
    print(f"Completed for {folder[i]}.")
    filter.clear()
    mse_calc.clear()
    psnr_calc.clear()
    uqi_calc.clear()
    scc_calc.clear()
    rase_calc.clear()
    ssim_calc.clear()
    noise_name.clear()
    pic_name.clear()

