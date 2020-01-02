import cv2
import numpy as np
from utils import show_mag_and_phase


def main():
    # read image
    img_el = cv2.imread('data/elephant.jpeg', 0)

    # get magnitude and phase using fourier transform
    f = np.fft.fft2(img_el)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift))
    phase = np.angle(fshift)

    # show magnitude and phase
    show_mag_and_phase(img_el, magnitude, phase)
    cv2.imwrite('data/elephant_magnitude.png', magnitude)
    cv2.imwrite('data/elephant_phase.png', phase)


if __name__ == '__main__':
    main()
