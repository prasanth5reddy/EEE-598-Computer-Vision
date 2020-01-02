import numpy as np


def convert_to_double(img):
    return img.astype(np.double)


def create_mask(shape):
    # takes shape of the image
    # return left blank and right white
    return np.concatenate((np.zeros((shape[0], shape[1] // 2, 3)),
                           np.ones((shape[0], shape[1] - shape[1] // 2, 3))),
                          axis=1)
