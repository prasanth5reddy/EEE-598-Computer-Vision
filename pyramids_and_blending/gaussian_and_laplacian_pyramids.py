import cv2
import matplotlib.pyplot as plt


def pyr_down(img):
    # perform gaussian blur on image
    blur = cv2.GaussianBlur(img, (25, 25), 5)
    # returns the dowsample image of blur image
    return cv2.resize(blur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)


def pyr_up(img):
    # returns the upsample image
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def show_pyr(pyrs):
    # shows the multiple pyramid images
    fig, axes = plt.subplots(nrows=1, ncols=PYR_DEPTH)
    for pyr, ax in zip(pyrs, axes):
        ax.imshow(pyr)
    plt.show()


def gaus_pyr(img, PYR_DEPTH=6):
    # takes an image and depth
    # returns a list of gaussian pyramids

    # first image is itself
    pyr = [img]
    for i in range(PYR_DEPTH - 1):
        img = pyr_down(img)
        pyr.append(img)
    return pyr


def lap_pyr(img, PYR_DEPTH=6):
    # takes an image and depth
    # returns a list of laplace pyramids

    # first image is img - gaussian blur
    lap = [img - cv2.GaussianBlur(img, (25, 25), 5)]
    for i in range(PYR_DEPTH - 2):
        # get next down image
        g_pyr = pyr_down(img)
        # add laplace pyramid to the list
        lap.append(g_pyr - cv2.GaussianBlur(g_pyr, (25, 25), 5))
        # change img to downsample gaussian pyr
        img = g_pyr
    lap.append(pyr_down(img))
    return lap


def cons_from_lap(pyrs, PYR_DEPTH=6):
    # takes laplace pyramids and depth
    # returns reconstructed image
    img = pyrs[PYR_DEPTH - 1]
    for i in range(PYR_DEPTH - 2, -1, -1):
        # upsample the lowest pyramid
        upsample = pyr_up(img)
        next_pyr = pyrs[i]
        # find min height and width to perform good addition
        h, w = min(upsample.shape[0], next_pyr.shape[0]), min(upsample.shape[1], next_pyr.shape[1])
        # add upsampled image and next pyramid
        img = upsample[:h, :w, :] + next_pyr[:h, :w, :]
    return img


def main():
    # read the image
    img_el = cv2.imread('data/elephant.jpeg', 1)
    # convert to RGB space
    img_el = cv2.cvtColor(img_el, cv2.COLOR_BGR2RGB)

    # Gaussian pyramids
    gaus_pyrs = gaus_pyr(img_el)
    for pyr in gaus_pyrs:
        print(pyr.shape)
        plt.imshow(pyr)
        plt.title('Gaussian Pyramids')
        plt.show()

    # Laplacian pyramids
    lap_pyrs = lap_pyr(img_el)
    # show_pyr(lap_pyrs)
    for pyr in lap_pyrs:
        print(pyr.shape)
        plt.imshow(pyr)
        plt.title('Laplace Pyramids')
        plt.show()

    # reconstruct img from laplacian pyramids
    img_back = cons_from_lap(lap_pyrs)
    # show reconstructed image
    plt.imshow(img_back)
    plt.title('Image reconstructed from laplace pyramids')
    plt.show()


if __name__ == '__main__':
    main()
