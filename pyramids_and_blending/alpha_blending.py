import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import convert_to_double, create_mask


def alpha_blending(img_1, img_2, mask):
    # takes two images and mask
    # returns alpha blending image

    # perform gaussian blur on mask before blending
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return ((1 - mask) * img_1).astype(np.uint8) + (mask * img_2).astype(np.uint8)


def main():
    # read images
    apple, orange = cv2.imread('data/apple.jpeg', 1), cv2.imread('data/orange.jpeg', 1)
    # convert images to double precision
    apple, orange = convert_to_double(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)), convert_to_double(
        cv2.cvtColor(orange, cv2.COLOR_BGR2RGB))
    # create mask
    mask = create_mask(apple.shape)

    # perform alpha blending
    alpha_blend = alpha_blending(apple, orange, mask)
    # display alpha blend image
    plt.imshow(alpha_blend)
    plt.title('Alpha blending')
    plt.show()
    # save alpha blend image
    cv2.imwrite('data/alpha_blend.png', cv2.cvtColor(alpha_blend, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
