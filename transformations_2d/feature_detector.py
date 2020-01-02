import cv2
from harris_corners import HarrisCorners


def main():
    # read original and rotated images
    quad_img, rotated_img = cv2.imread('data/original_quad.jpg', 1), cv2.imread('data/rotated_quad.jpg', 1)
    # compute harris corners from scratch
    quad_harris_corners = HarrisCorners(quad_img, window_size=7, k=0.04, threshold=0.05)
    quad_corners_indices = quad_harris_corners.compute_corners()
    print(f'Original quadrilateral detected points : {quad_corners_indices}')
    quad_harris_corners.show_corners('Quadrilateral Original')

    rot_harris_corners = HarrisCorners(rotated_img, window_size=7, k=0.04, threshold=0.15)
    rot_corners_indices = rot_harris_corners.compute_corners()
    print(f'Rotated quadrilateral detected points : {rot_corners_indices}')
    rot_harris_corners.show_corners('Quadrilateral Rotated')


if __name__ == '__main__':
    main()
