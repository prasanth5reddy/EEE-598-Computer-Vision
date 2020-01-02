import cv2
import numpy as np
from utils import inverse_transform, show_filter_plots, circle_mask


def low_pass_filter(img_el, magnitude, fshift):
    # performs series of operations for low pass filter
    # and plots the different images

    # create circle mask for low pass filter
    # for this image 40 is taken as threshold circle radius
    mask = circle_mask(img_el.shape, 40)
    # magnitude after filtering
    low_pass = magnitude * mask
    # get back original image after filtering
    img_el_back = inverse_transform(fshift * mask)
    # show low pass filter plots
    show_filter_plots(magnitude, mask, low_pass, img_el_back)
    return img_el_back


def high_pass_filter(img_el, magnitude, fshift):
    # performs series of operations for high pass filter
    # and plots the different images

    # create circle mask for high pass filter
    # for this image 150 is taken as threshold circle radius
    mask = circle_mask(img_el.shape, 150)
    # magnitude after filtering
    high_pass = magnitude * (1 - mask)
    # get back original image after filtering
    img_el_back = inverse_transform(fshift * (1 - mask))
    # show high pass filter plots
    show_filter_plots(magnitude, mask, high_pass, img_el_back, f_name='high')
    return img_el_back


def band_pass_filter(img_el, magnitude, fshift):
    # performs series of operations for high pass filter
    # and plots the different images

    # create circle mask for band pass filter
    # for this image two thresholds 250 and 100 are taken
    mask = circle_mask(img_el.shape, 250) - circle_mask(img_el.shape, 100)
    # magnitude after filtering
    high_pass = magnitude * mask
    # get back original image after filtering
    img_el_back = inverse_transform(fshift * mask)
    # show band pass filter plots
    show_filter_plots(magnitude, mask, high_pass, img_el_back, f_name='band')
    return img_el_back


def main():
    # read image
    img_el = cv2.imread('data/elephant.jpeg', 0)

    # get magnitude and phase using fourier transform
    f = np.fft.fft2(img_el)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift))
    phase = np.angle(fshift)

    # apply low pass filter
    low_pass_img = low_pass_filter(img_el, magnitude, fshift)
    # save low pass filtered image
    cv2.imwrite('data/elephant_lowpass.png', low_pass_img)

    # apply high pass filter
    high_pass_img = high_pass_filter(img_el, magnitude, fshift)
    # save low pass filtered image
    cv2.imwrite('data/elephant_highpass.png', high_pass_img)

    # apply band pass filter
    band_pass_img = band_pass_filter(img_el, magnitude, fshift)
    # save band pass filtered image
    cv2.imwrite('data/elephant_bandpass.png', band_pass_img)


if __name__ == '__main__':
    main()
