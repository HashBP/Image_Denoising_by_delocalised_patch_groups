import numpy as np
import cv2
from skimage.util import random_noise
from env import *

def add_all_noises():
    img = cv2.imread(ORIGINAL_IMG_NAME)

    # ///////////////////////////////////////// NOISING ////////////////////////////////////////////////

    def sp_noise(img, prob):
        noise_img = random_noise(img, mode='s&p', amount=prob)
        noise_img = np.array(255*noise_img, dtype='uint8')
        return noise_img

    def poisson_noise(image, prob):
        noise = np.random.poisson(50, image.shape)
        output = image+noise*prob
        return output

    def speckle_noise(image, prob):
        gauss = np.random.normal(0, prob, image.size)
        gauss = gauss.reshape(
            image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        noise = image + image * gauss
        return noise

    def gaussian_noise(image, prob):
        gauss = np.random.normal(0, prob, image.size)
        gauss = gauss.reshape(
            image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        img_gauss = cv2.add(image, gauss)
        return img_gauss

    for i in range(len(NOISE_MAGNITUDE)):
        sp_noisy_img = sp_noise(img, NOISE_MAGNITUDE[i])
        gaussian_noisy_img = gaussian_noise(img, NOISE_MAGNITUDE[i])
        poisson_noisy_img = poisson_noise(img, NOISE_MAGNITUDE[i])
        speckle_noisy_img = speckle_noise(img, NOISE_MAGNITUDE[i])
# ////////////////////////////////SAVING NOISED IMAGE //////////////////////////////////////////////////

        cv2.imwrite(f"noisy_images/sp_noise{int(NOISE_MAGNITUDE[i]*100)}.jpg", sp_noisy_img)
        cv2.imwrite(f"noisy_images/gaussian_noise{int(NOISE_MAGNITUDE[i]*100)}.jpg", gaussian_noisy_img)
        cv2.imwrite(f"noisy_images/speckle_noise{int(NOISE_MAGNITUDE[i]*100)}.jpg", speckle_noisy_img)
        cv2.imwrite(f"noisy_images/poisson_noise{int(NOISE_MAGNITUDE[i]*100)}.jpg", poisson_noisy_img)
        print(f"Done Added {NOISE_MAGNITUDE[i]*100}% noises to the Images..")

add_all_noises()