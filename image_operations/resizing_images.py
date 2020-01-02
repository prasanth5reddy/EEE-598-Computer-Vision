import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)


def main():
    # read image
    img_el = cv2.imread('data/elephant.jpeg', 1)
    # convert to RGB space
    img_el = cv2.cvtColor(img_el, cv2.COLOR_BGR2RGB)

    # downsample image by 10x in width and height
    ds_img = cv2.resize(img_el, None, fx=0.1, fy=0.1)
    # display downsampled image
    plt.imshow(ds_img)
    plt.title('Downsampled Elephant')
    plt.show()
    # write downsampled image
    cv2.imwrite('data/elephant_10xdown.png', cv2.cvtColor(ds_img, cv2.COLOR_RGB2BGR))

    # upsample the downsample image using nearest neighbour method
    us_img_nn = cv2.resize(ds_img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    # upsample the downsample image using bicubic method
    us_img_bc = cv2.resize(ds_img, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
    plt.show()

    # write both upsampled images
    cv2.imwrite('data/elephant_10xup_nearest_neighbour.png', cv2.cvtColor(us_img_nn, cv2.COLOR_RGB2BGR))
    cv2.imwrite('data/elephant_10xup_bicubic.png', cv2.cvtColor(us_img_bc, cv2.COLOR_RGB2BGR))

    # absolute difference between ground truth image and upsampled images
    ad_img_nn = cv2.absdiff(img_el, us_img_nn)
    ad_img_bc = cv2.absdiff(img_el, us_img_bc)
    # write both absolute difference images
    cv2.imwrite('data/elephant_10xup_absdiff_nearest_neighbour.png', cv2.cvtColor(ad_img_nn, cv2.COLOR_RGB2BGR))
    cv2.imwrite('data/elephant_10xup_absdiff_bicubic.png', cv2.cvtColor(ad_img_bc, cv2.COLOR_RGB2BGR))

    # print sum of pixels in the difference image for both methods
    print('Nearest neighbour : ', np.sum(ad_img_nn), '\nBicubic : ', np.sum(ad_img_bc))
    # After looking at the sum values, bicubic method caused less error in upsampling


if __name__ == '__main__':
    main()
