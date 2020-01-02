import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)


def main():
    # read image
    img_el = cv2.imread('data/elephant.jpeg', 1)
    # convert to RGB space
    img_el = cv2.cvtColor(img_el, cv2.COLOR_BGR2RGB)
    # add 256 to each pixel
    img_el = img_el + 256
    # change to uint8
    img_el = img_el.astype(np.uint8)
    # display image
    plt.imshow(img_el)
    plt.title('Elephant + 256')
    plt.show()
    # After adding 256 to every pixel the image datatype changed to uint16 to
    # accomodate maximum number i.e 255 + 255. But, after converting to uint8
    # datatype the pixel values changed back to normal range of 0-255.
    # Since we have added 256 and divided by it, the pixel values didn't change.
    # e.g: (95 + 256) % 256 = 95

    # add 256 using opencv
    # split into multiple channels
    b, g, r = cv2.split(img_el)
    # add 256 to every pixel in each channel
    b = cv2.add(b, 256)
    g = cv2.add(g, 256)
    r = cv2.add(r, 256)
    # merge the three channels
    mrgd_img = cv2.merge((b, g, r))
    # display merged image
    plt.imshow(mrgd_img)
    plt.title('Elephant cv2.add 256')
    plt.show()
    # The image formed at the end is white with all pixel values as 255. This is
    # because OpenCV addition is saturated operation, i.e. after adding, if values
    # are more than 255 it will get rounded to 255. Whereas, numpy addition is
    # modulo operation with  % 256 at the end.


if __name__ == '__main__':
    main()
