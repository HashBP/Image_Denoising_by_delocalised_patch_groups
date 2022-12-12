from sewar.full_ref import mse, psnr, ssim,uqi,scc,rase
import cv2
from tabulate import tabulate
from env import *
import pandas as pd
import os

percent_n=[]
mse_calc=[]
psnr_calc=[]
uqi_calc=[]
scc_calc=[]
rase_calc=[]
ssim_calc=[]

def compare_images(path,percent_noise):
    # l1 = os.listdir("itr_filtered_images/Blue_Channel")
    # l2 = os.listdir("itr_filtered_images/Green_Channel")
    # l3 = os.listdir("itr_filtered_images/Red_Channel")
    # iter = min(len(l1), len(l2), len(l3))
    # sample_img = []
    # for i in range(iter-1):
    #     sample_img.append(cv2.imread(f"merged_images/Merged_itr{i}.jpg"))
    # my_data.append(row)
    # print(tabulate(my_data, headers=head, tablefmt="grid"))
    # row = []
    # my_data = []
    # head = ["Noise Percent", "MSE⬇️", "UQI", "PSNR⬆️", "SSIM⬆️","SCC","RASE"]
    # col = percent_noise

    org_img = cv2.imread(ORIGINAL_IMG_NAME)
    comp_img = cv2.imread(path)
    m = mse(org_img, comp_img)
    u = uqi(org_img, comp_img)
    s = scc(org_img, comp_img)
    r = rase(org_img, comp_img)
    p = psnr(org_img, comp_img)
    ss = ssim(org_img, comp_img)[0]
    percent_n.append(percent_noise)
    mse_calc.append(m)
    uqi_calc.append(u)
    psnr_calc.append(p)
    ssim_calc.append(ss)
    scc_calc.append(s)
    rase_calc.append(r)

def make_csv(noise_name,image_name):
    dict={
        "Noise(in %)":percent_n,
        "MSE⬇️":mse_calc,
        "RASE⬇️":rase_calc,
        "UQI⬆️":uqi_calc,
        "PSNR⬆️":psnr_calc,
        "SSIM⬆️":ssim_calc,
        "SCC⬆️":scc_calc
    }
    df=pd.DataFrame(dict)
    df.to_csv(f"Metric_Data/{image_name}_Metrics/Metric_{noise_name}.csv", index=False)
    
