import numpy as np
import matplotlib.pyplot as plt


def circle_mask(shape, r):
    # takes image shape and radius of circle that needs to be filled
    # return circle mask image
    c = np.zeros(shape)
    x, y = shape[0] // 2, shape[1] // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (x - i) ** 2 + (y - j) ** 2 < r ** 2:
                c[i][j] = 1
    return c


def show_mag_and_phase(img, mag, ph):
    # takes image, magnitude and phase array and shows those plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    # original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Image')
    # magnitude image
    axes[1].imshow(mag, cmap='gray')
    axes[1].set_title('Magnitude')
    # phase image
    axes[2].imshow(ph, cmap='gray')
    axes[2].set_title('Phase')
    plt.show()


def inverse_transform(fshift):
    # takes fshift and performs inverse transform
    # returns image back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def show_filter_plots(mag, mask, mag_filter, img_back, f_name='low'):
    # displays the image magnitude, mask, filtered magnitude,
    # filtered image back and filter name
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    # magnitude before filter
    axes[0].imshow(mag, cmap='gray')
    axes[0].set_title('Magnitude before filter')
    # show filter
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Circle mask')
    # magnitude after filter
    axes[2].imshow(mag_filter, cmap='gray')
    axes[2].set_title('Magnitude after filtering')
    plt.show()
    # image after filter
    plt.imshow(img_back, cmap='gray')
    plt.title('Image after ' + f_name + ' pass filter')
    plt.show()
