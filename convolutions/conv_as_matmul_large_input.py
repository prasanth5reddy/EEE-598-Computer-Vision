import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import time
from conv_as_matmul import zero_pad, toeplitz_matrices, doubly_blocked_toeplitz_matrix


def conv2d_elephant(I, K):
    # idea is to split image matrix into multiple patches and stitch together

    s = time.time()
    # find output height and weight
    h, w = I.shape[:2]
    h_k, w_k = K.shape[:2]
    h_o, w_o = h + h_k - 1, w + w_k - 1
    conv2d_el = np.zeros((h_o, w_o))

    # patch parameters
    PATCH_SIZE_INPUT = 100, 100
    PATCH_SIZE_OUTPUT = PATCH_SIZE_INPUT[0] + h_k - 1, PATCH_SIZE_INPUT[1] + w_k - 1
    # zero pad kernel
    k_zero_pad = zero_pad(K, PATCH_SIZE_OUTPUT)
    # computer toeplitz matrices for all the rows of zero padded kernel
    toep_mats = toeplitz_matrices(k_zero_pad, PATCH_SIZE_INPUT[1])
    # compute doubly blocked toeplitz matrix
    db_toep_mat = doubly_blocked_toeplitz_matrix(toep_mats, PATCH_SIZE_INPUT[0])

    for i in range(h_o // PATCH_SIZE_INPUT[0]):
        for j in range(w_o // PATCH_SIZE_INPUT[1]):
            patch = I[i * PATCH_SIZE_INPUT[0]:(i + 1) * PATCH_SIZE_INPUT[0],
                    j * PATCH_SIZE_INPUT[1]: (j + 1) * PATCH_SIZE_INPUT[1]]

            # convert input patch into vector
            patch_vec = patch.reshape(PATCH_SIZE_INPUT[0] * PATCH_SIZE_INPUT[1])

            # multiply doubly blocked toeplitz matrix with reshaped input
            patch_res_vec = np.matmul(db_toep_mat, patch_vec)

            # reshape result vector to matrix
            conv2d_patch = patch_res_vec.reshape((PATCH_SIZE_OUTPUT[0], PATCH_SIZE_OUTPUT[1]))

            conv2d_el[i * PATCH_SIZE_INPUT[0]:(i + 1) * PATCH_SIZE_INPUT[0],
            j * PATCH_SIZE_INPUT[1]: (j + 1) * PATCH_SIZE_INPUT[1]] = conv2d_patch[1:-1, 1:-1]

    conv_scipy = signal.convolve2d(I, K, 'full')
    mse_el = np.mean((conv2d_el - conv_scipy) ** 2).item()
    run_time_el = time.time() - s
    return conv2d_el, run_time_el, mse_el


def main():
    # read elephant image
    img_el = cv2.imread('data/elephant.jpeg', 0)
    H = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]])

    # elephant convolution as matrix multiplication
    conv2d_el, run_time_el, mse_el = conv2d_elephant(img_el, H)
    print('convolution output\n', conv2d_el, '\nrun time in sec\n', run_time_el, '\nmean Square Error\n', mse_el)

    # show convoluted elephant image
    plt.imshow(conv2d_el, cmap='gray')
    plt.title('Elephant image after convolution')
    plt.show()

    # write convoluted image
    cv2.imwrite('data/elephant_conv.png', conv2d_el)


if __name__ == '__main__':
    main()
