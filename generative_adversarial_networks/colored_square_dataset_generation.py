import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_random_square():
    # set up point
    x, y = np.random.normal(32, 8, 2)
    x, y = int(x), int(y)
    # set up size
    s = np.random.normal(8, 16, 1)
    s = int(s[0])
    # set up color
    b, g, r = np.random.randint(0, 256, 1), np.random.randint(0, 256, 1), np.random.randint(0, 256, 1)
    color = (int(b[0]), int(g[0]), int(r[0]))
    # create square
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(img, (x, y), (x + s, y + s), color, -1)
    # set up orientation
    angle = np.random.randint(0, 46, 1)
    rotation_matrix = cv2.getRotationMatrix2D((32, 32), angle, 1)
    # get the rotated image through warp affine
    return cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))


def create_square_dataset(path, n):
    for i in range(1, n + 1):
        name = ['0'] * (len(str(n)) - len(str(i))) + [str(i)]
        name = ''.join(name) + '.jpg'
        plt.imsave(path + '/img_square/' + name, create_random_square())
        print(f'created image {name}')


def main():
    path = '/Users/prasanth/Academics/ASU/FALL_2019/EEE_598_CIU/GAN/data/Square'
    no_of_images = 100000
    create_square_dataset(path, no_of_images)


if __name__ == '__main__':
    main()
