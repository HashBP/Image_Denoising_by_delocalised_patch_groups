from add_all_noise import *
from compare_images_csv import *
from filter_start import *
import os
for l in range(len(img)):
    os.makedirs(f"Metric_Data/{img[k]}_Metrics")
    add_all_noises()
    start_filter() 
    k+=1