import cv2
import matplotlib.pyplot as plt

plt.interactive(False)


def main():
    # load image
    img_el = cv2.imread('data/elephant.jpeg', 1)
    # cv2.imshow('Elephant', img_el)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # display image
    plt.imshow(img_el)
    plt.title('Elephant OpenCV')
    plt.show()

    # write image
    cv2.imwrite('data/elephant_opencv.png', img_el)

    # convert to RGB and display image
    rgb_img_el = cv2.cvtColor(img_el, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img_el)
    plt.title('Elephant Matplotlib')
    plt.show()

    # write rgb converted image
    cv2.imwrite('data/elephant_matplotlib.png', rgb_img_el)

    # read grayscale image
    gray_img_el = cv2.imread('data/elephant.jpeg', 0)

    # display grayscale image
    plt.imshow(gray_img_el, cmap='gray')
    plt.title('Elephant Grayscale')
    plt.show()

    # write grayscale image
    cv2.imwrite('data/elephant_gray.png', gray_img_el)


if __name__ == '__main__':
    main()
