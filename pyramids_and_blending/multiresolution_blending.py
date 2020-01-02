import matplotlib.pyplot as plt
import cv2
from utils import convert_to_double, create_mask
from gaussian_and_laplacian_pyramids import gaus_pyr, lap_pyr, cons_from_lap
from direct_blending import direct_blending


def multiresolution_blending(img_1, img_2, mask):
    # takes two images and mask
    # returns multiresolution blending image

    # fix pyramid depths
    pyr_depth = 6
    # get gaussian pyramids of mask
    mask_pyrs = gaus_pyr(mask, pyr_depth)
    # get laplace pyramids of two images
    img_1_pyrs = lap_pyr(img_1, pyr_depth)
    img_2_pyrs = lap_pyr(img_2, pyr_depth)

    # create a list of blended pyrs
    blend_pyrs = []
    for i in range(pyr_depth - 1, -1, -1):
        # perform direct blending on each pyramid
        blend_pyrs.append(direct_blending(img_1_pyrs[i], img_2_pyrs[i], mask_pyrs[i]))

    # reverse the blended pyramids list
    blend_pyrs.reverse()
    # construct image from laplace
    return cons_from_lap(blend_pyrs, pyr_depth)


def main():
    # read images
    apple, orange = cv2.imread('data/apple.jpeg', 1), cv2.imread('data/orange.jpeg', 1)
    # convert images to double precision
    apple, orange = convert_to_double(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)), convert_to_double(
        cv2.cvtColor(orange, cv2.COLOR_BGR2RGB))
    # create mask
    mask = create_mask(apple.shape)

    # perform multiresolution blending
    multires_blend = multiresolution_blending(apple, orange, mask)
    # display multiresolution blend image
    plt.imshow(multires_blend)
    plt.title('Multiresolution blending with apple left')
    plt.show()
    # save multiresolution blend image
    cv2.imwrite('data/multires_blend_apponge.png', cv2.cvtColor(multires_blend, cv2.COLOR_RGB2BGR))

    # perform multiresolution blending
    multires_blend = multiresolution_blending(orange, apple, mask)
    # display multiresolution blend image
    plt.imshow(multires_blend)
    plt.title('Multiresolution blending with orange left')
    plt.show()
    # save multiresolution blend image
    cv2.imwrite('data/multires_blend_orapple.png', cv2.cvtColor(multires_blend, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
