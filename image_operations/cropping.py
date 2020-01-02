import cv2
import matplotlib.pyplot as plt

plt.interactive(False)


def main():
    img_el = cv2.imread('data/elephant.jpeg', 1)

    # crop image
    crop_img_el = img_el[350:940, 100:550]

    # display baby elephant image
    plt.imshow(cv2.cvtColor(crop_img_el, cv2.COLOR_BGR2RGB))
    plt.title('Baby Elephant')
    plt.show()

    # write baby elephant image
    cv2.imwrite('data/babyelephant.png', crop_img_el)


if __name__ == '__main__':
    main()
