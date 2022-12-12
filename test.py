import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

img =img_as_float(io.imread("noisy_images/gaussian_noise80.jpg"))
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(
    img, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

cv2.imshow("Original", img)
cv2.imshow(f"{sigma_est}", denoise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
