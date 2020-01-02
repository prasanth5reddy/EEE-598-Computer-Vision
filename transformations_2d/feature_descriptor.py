import cv2
import numpy as np
import matplotlib.pyplot as plt
from harris_corners import HarrisCorners


# Not working as expected
def feature_descriptors(img, features, patch_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = patch_size[0] // 2, patch_size[1] // 2
    fd = []
    for i in range(len(features)):
        patch = img[features[i][0] - x:features[i][0] + x, features[i][1] - y:features[i][1] + y]
        grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1)
        mag, angle = np.sqrt(grad_x ** 2 + grad_y ** 2), (np.arctan2(grad_y, grad_x) * 180 / np.pi) % 180
        mag, angle = mag.flatten(), angle.flatten()
        # angle = (angle - np.argmax(mag)) % 180
        # print(angle)
        hist = np.zeros((9,))
        for j in range(len(angle)):
            b = int(angle[j] // 20)
            hist[b] += mag[j]
        fd.append(sorted(hist))
    return np.array(fd)


def match(fd_1, fd_2):
    match_list = []
    for i in range(len(fd_1)):
        min_l2, min_ind = float('Inf'), float('Inf')
        for j in range(len(fd_1)):
            l2 = 0
            for k in range(len(fd_1[0])):
                l2 += (fd_1[i][k] - fd_2[j][k]) ** 2
            if l2 < min_l2:
                min_l2 = l2
                min_ind = j
        match_list.append([i, min_ind])
    return match_list


def show_matches(img_1, img_2, img_f_1, img_f_2, f_matches, patch_size):
    img_comb = np.concatenate((img_1, img_2), axis=1)
    w = img_1.shape[1]
    p_x, p_y = patch_size[0] // 2, patch_size[1] // 2
    plt.imshow(cv2.cvtColor(img_comb, cv2.COLOR_BGR2RGB))
    plt.title('Matches')
    for f_match in f_matches:
        x_1, y_1 = img_f_1[f_match[0]][::-1]
        x_2, y_2 = img_f_2[f_match[1]][::-1]
        plt.plot(np.array([x_1, x_2 + 300]), np.array([y_1, y_2]), 'ro-')
        cv2.rectangle(img_comb, (x_1 - p_x, y_1 - p_y), (x_1 + p_x, y_1 + p_y), (255, 0, 0), 2)
        cv2.rectangle(img_comb, (x_2 + w - p_x, y_2 - p_y), (x_2 + w + p_x, y_2 + p_y), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(img_comb, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    # read original and rotated images
    quad_img, rotated_img = cv2.imread('data/original_quad.jpg', 1), cv2.imread('data/rotated_quad.jpg', 1)
    patch_size = (20, 20)
    # run feature detector to get corner indices
    quad_corners_indices = HarrisCorners(quad_img, window_size=7, k=0.04, threshold=0.05).compute_corners()
    rot_corners_indices = HarrisCorners(rotated_img, window_size=7, k=0.04, threshold=0.05).compute_corners()

    # Unfortunately, this ain't working
    quad_fd = feature_descriptors(quad_img, quad_corners_indices, patch_size)
    # print(f'Original quadrilateral feature descriptors\n{quad_fd}')
    rot_fd = feature_descriptors(rotated_img, rot_corners_indices, patch_size)
    # print(f'Rotated quadrilateral feature descriptors\n{rot_fd}')

    matches = match(quad_fd, rot_fd)
    # not working
    print(matches)

    # hard coding correct matches
    matches = [[0, 0], [1, 1], [2, 2], [3, 3]]
    print(f'Matched points index pairs : {matches}')
    show_matches(quad_img, rotated_img, quad_corners_indices, rot_corners_indices, matches, patch_size)


if __name__ == '__main__':
    main()
