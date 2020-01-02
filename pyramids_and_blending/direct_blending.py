import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import convert_to_double, create_mask


def direct_blending(img_1, img_2, mask):
    # takes two images and mask
    # return direct blending image
    return ((1 - mask) * img_1).astype(np.uint8) + (mask * img_2).astype(np.uint8)


def main():
    # read apple image
    apple = cv2.imread('data/apple.jpeg', 1)
    apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
    # read orange image
    orange = cv2.imread('data/orange.jpeg', 1)
    orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)
    # convert apple and orange to double precision
    apple, orange = convert_to_double(apple), convert_to_double(orange)

    # create mask
    mask = create_mask(apple.shape)
    # display mask
    plt.imshow(mask)
    plt.title('Mask')
    plt.show()

    # perform direct blending
    direct_blend = direct_blending(apple, orange, mask)
    # display direct blend image
    plt.imshow(direct_blend)
    plt.title('Direct blending')
    plt.show()
    # save direct blend image
    cv2.imwrite('data/direct_blend.png', cv2.cvtColor(direct_blend, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
