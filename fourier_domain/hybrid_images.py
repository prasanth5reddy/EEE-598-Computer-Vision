import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import circle_mask, inverse_transform


def hybrid_image(fs_1, fs_2, img_1):
    # takes fshift of two images
    # returns hybrid image

    # create circle mask for low pass filter
    # here threshold is taken as 6
    mask = circle_mask(img_1.shape, 6)
    # get back image 1 after filtering
    img_1_back = inverse_transform(fs_1 * mask)
    # get back image 2 after filtering
    img_2_back = inverse_transform(fs_2 * (1 - mask))
    # add the two images and return
    return cv2.add(img_1_back, img_2_back)


def main():
    # read images
    # for this example grand canyon and niagara falls images are taken
    img_1 = cv2.imread('data/gc.jpg', 0)
    img_2 = cv2.imread('data/nf.jpg', 0)

    # perform fourier transform on both images
    f_1 = np.fft.fft2(img_1)
    fshift_1 = np.fft.fftshift(f_1)
    f_2 = np.fft.fft2(img_2)
    fshift_2 = np.fft.fftshift(f_2)

    # perform hybrid operations on both images
    hy_img = hybrid_image(fshift_1, fshift_2, img_1)
    # display hybrid image
    plt.imshow(hy_img, cmap='gray')
    plt.title('Hybrid image')
    plt.show()
    # save hybrid image
    cv2.imwrite('data/gc_nf_hyb.png', hy_img)


if __name__ == '__main__':
    main()
