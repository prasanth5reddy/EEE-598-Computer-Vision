import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, title='Figure', color=None):
    if not color:
        plt.imshow(img)
    elif color == 'gray':
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, color))
    plt.title(title)
    plt.show()


# returns an image with quadrilateral
def construct_quadrilateral(img_size, pts, color):
    # Create an image with black background
    h, w = img_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # fill the quadrilateral with white
    cv2.fillPoly(img, [pts], color)
    return img


# returns a translated image
def translate_image(img, pts):
    h, w = img.shape[:2]
    # create translate matrix
    translate_matrix = np.float32([[1, 0, pts[0]], [0, 1, pts[1]]])
    # get the translated image through warp affine
    return cv2.warpAffine(img, translate_matrix, (w, h))


# returns a rotated image
def rotate_image(img, angle):
    h, w = img.shape[:2]
    # create rotation matrix with centre of image as point of rotation
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # get the rotated image through warp affine
    return cv2.warpAffine(img, rotation_matrix, (w, h))


def main():
    # create an image with quadrilateral
    IMG_HEIGHT, IMG_WIDTH = 300, 300
    # set some random points for the quadrilateral
    random_pts = np.array([[50, 60], [100, 190], [200, 160], [220, 100]])
    quad_img = construct_quadrilateral((IMG_HEIGHT, IMG_WIDTH), random_pts, (255, 255, 255))
    # show the constructed image
    show_image(quad_img, title='Quadrilateral image', color=cv2.COLOR_BGR2RGB)
    # save the quadrilateral image
    cv2.imwrite('data/original_quad.jpg', quad_img)

    # Translate the image by (30, 100)
    TRANSLATE_X, TRANSLATE_Y = 30, 100
    translated_img = translate_image(quad_img, (TRANSLATE_X, TRANSLATE_Y))

    # Rotate the image around origin (150, 150)
    ROTATION_ANGLE = 45
    rotated_img = rotate_image(translated_img, ROTATION_ANGLE)

    # show the rotated image
    show_image(rotated_img, title='Rotated image', color=cv2.COLOR_BGR2RGB)
    # save the rotated image
    cv2.imwrite('data/rotated_quad.jpg', rotated_img)


if __name__ == '__main__':
    main()
