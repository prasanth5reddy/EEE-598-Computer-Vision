import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import show_mag_and_phase, inverse_transform


def swap_phases(f_shift, ph):
    return np.multiply(np.abs(f_shift), np.exp(1j * ph))


def main():
    # read images
    # for this example grand canyon and niagara falls images are taken
    img_1 = cv2.imread('data/gc.jpg', 0)
    img_2 = cv2.imread('data/nf.jpg', 0)

    # get magnitude and phase for image 1
    f_1 = np.fft.fft2(img_1)
    fshift_1 = np.fft.fftshift(f_1)
    mag_1 = 20 * np.log(np.abs(fshift_1))
    phase_1 = np.angle(fshift_1)

    # get magnitude and phase for image 2
    f_2 = np.fft.fft2(img_2)
    fshift_2 = np.fft.fftshift(f_2)
    mag_2 = 20 * np.log(np.abs(fshift_2))
    phase_2 = np.angle(fshift_2)

    # show magnitude and phase for both images
    show_mag_and_phase(img_1, mag_1, phase_1)
    show_mag_and_phase(img_2, mag_2, phase_2)

    # perform phase swap and inverse transform to get image
    img_mag_1_phase_2 = inverse_transform(swap_phases(fshift_1, phase_2))
    plt.imshow(img_mag_1_phase_2, cmap='gray')
    plt.title('Grand canyon magnitude with Niagara phase')
    plt.show()
    # save swapped phase image
    cv2.imwrite('data/gc_mag_nf_phase.png', img_mag_1_phase_2)

    # perform phase swap and inverse transform to get image
    img_mag_2_phase_1 = inverse_transform(swap_phases(fshift_2, phase_1))
    plt.imshow(img_mag_2_phase_1, cmap='gray')
    plt.title('Niagara magnitude with Grand Canyon phase')
    plt.show()
    # save swapped phase image
    cv2.imwrite('data/nf_mag_gc_phase.png', img_mag_2_phase_1)


if __name__ == '__main__':
    main()
