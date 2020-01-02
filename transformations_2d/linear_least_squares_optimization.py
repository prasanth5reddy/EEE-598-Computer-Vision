import cv2
import numpy as np
from harris_corners import HarrisCorners


def lls_opt(pts_1, pts_2, pt_match):
    # create empty list for matrix A
    matrix_a = []
    for i in range(len(pt_match)):
        # iterate through each pair of points
        # get the matching point pairs from each of the images
        pt_pair = pts_1[pt_match[i][0]][::-1], pts_2[pt_match[i][1]][::-1]
        # Extract the coordinates
        (x_1, y_1), (x_2, y_2) = pt_pair[0], pt_pair[1]
        # form linear equations from the points
        a_x = [-x_1, -y_1, -1, 0, 0, 0, x_2 * x_1, x_2 * y_1, x_2]
        a_y = [0, 0, 0, -x_1, -y_1, -1, y_2 * x_1, y_2 * y_1, y_2]
        # add the equations to the matrix A
        matrix_a.append(a_x)
        matrix_a.append(a_y)

    # create matrix A of type numpy array
    matrix_a = np.array(matrix_a)
    # apply singular value decomposition
    u, s, vh = np.linalg.svd(matrix_a)
    # reshape the final vector into h matrix
    matrix_h = vh[8].reshape(3, 3)
    # divide the matrix by last row, column value to get 1 at this place
    matrix_h = (matrix_h / matrix_h[2][2])
    return matrix_h


def main():
    # read original and rotated images
    quad_img, rotated_img = cv2.imread('data/original_quad.jpg', 1), cv2.imread('data/rotated_quad.jpg', 1)
    patch_size = (20, 20)
    # run feature detector to get corner indices
    quad_corners_indices = HarrisCorners(quad_img, window_size=7, k=0.04, threshold=0.05).compute_corners()
    rot_corners_indices = HarrisCorners(rotated_img, window_size=7, k=0.04, threshold=0.05).compute_corners()

    # set matches
    matches = [[0, 0], [1, 1], [2, 2], [3, 3]]
    # compute homogeneous matrix
    h_matrix = lls_opt(quad_corners_indices, rot_corners_indices, matches)
    print(f'H Matrix\n{np.round(h_matrix, 3)}')

    h_matrix_opencv, _ = cv2.findHomography(np.array([i[::-1] for i in quad_corners_indices][:4]),
                                            np.array([i[::-1] for i in rot_corners_indices][:4]))
    print(f'H Matrix computed by OpenCV\n{np.round(h_matrix_opencv, 3)}')

    # compute rotation and translation
    # To Do
    theta = int(np.arccos(round(h_matrix[0][0], 4)) * 180 / np.pi)
    t_x = int(round(h_matrix[0][2]))
    t_y = int(round(h_matrix[1][2]))
    print(f'theta : {theta}, t_x : {t_x}, t_y : {t_y}')


if __name__ == '__main__':
    main()
