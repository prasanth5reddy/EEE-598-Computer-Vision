import numpy as np
from scipy import signal
import time


def zero_pad(K, shape_o):
    # takes kernel and shape of output
    # return kernel with padded zeros at bottom and right to match with output size

    # get shapes of kernel and output
    h_k, w_k = K.shape[:2]
    h_o, w_o = shape_o[:2]
    # compute bottom zeroes
    bottom_zero = np.array([[0] * w_k] * (h_o - h_k))
    # compute right zeroes
    right_zero = np.array([[0] * (w_o - w_k)] * h_o)
    # pad zeros by concatenating with original kernel
    k_zero_pad = np.concatenate((np.concatenate((K, bottom_zero), axis=0), right_zero), axis=1)
    return k_zero_pad


def toeplitz(r, w):
    # takes a row and computes its toeplitz matrix based on number of columns w
    output = np.zeros((r.shape[0], w))
    for i in range(r.shape[0]):
        for j in range(w):
            if i >= j:
                output[i][j] = r[i - j]
    return output


def toeplitz_matrices(K, w):
    # takes a kernel and computes toeplitz matrix for each row
    h_k, o_k = K.shape[:2]
    # create list of topelitz matrices
    toep_mats = []
    for i in range(h_k):
        # compute toeplitz matrix and append to the list
        toep_mats.append(toeplitz(K[i], w))
    return np.array(toep_mats)


def doubly_blocked_toeplitz_matrix(toep_mats, h):
    # takes a list of topelitz matrices and input height h
    # return doubly toeplitz matrix

    # get number of, height and width of toeplitz matrices
    n_t, h_t, w_t = toep_mats.shape[0], toep_mats.shape[1], toep_mats.shape[2]
    # calculate output size
    db_h, db_w = n_t * h_t, w_t * h
    # initialize output matrix with zeros
    output = np.zeros((db_h, db_w))

    # fill in the toeplitz matrices at the right places
    for i in range(n_t):
        for j in range(h):
            if i >= j:
                output[i * h_t:(i + 1) * h_t, j * w_t:(j + 1) * w_t] = toep_mats[i - j]
    return output


def conv_as_matmul(I, K):
    # find output height and weight
    h, w = I.shape[:2]
    h_k, w_k = K.shape[:2]
    h_o, w_o = h + h_k - 1, w + w_k - 1
    # zero pad kernel
    k_zero_pad = zero_pad(K, (h_o, w_o))
    # computer toeplitz matrices for all the rows of zero padded kernel
    toep_mats = toeplitz_matrices(k_zero_pad, w)
    # compute doubly blocked toeplitz matrix
    db_toep_mat = doubly_blocked_toeplitz_matrix(toep_mats, h)
    # convert input matrix into vector
    I_vec = I.reshape(h * w)
    # multiply doubly blocked toeplitz matrix with reshaped input
    res_vec = np.matmul(db_toep_mat, I_vec)
    # reshape result vector to matrix
    conv = res_vec.reshape((h_o, w_o))
    # print('resultant matrix\n', conv)

    return conv


def conv2dmatrix(I, H):
    # start timer
    start = time.time()
    # perfrom convolution as matrix multiplication
    conv_matmul = conv_as_matmul(I, H)
    # get run time
    run_time = round(time.time() - start, 4)

    # get convolution from scipy library
    conv_scipy = signal.convolve2d(I, H, 'full')
    # calculate mse between two convolution results
    mse = np.mean((conv_matmul - conv_scipy) ** 2).item()
    return conv_matmul, run_time, mse


def main():
    I = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    H = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]])

    conv2d, run_time, mse = conv2dmatrix(I, H)
    print('convolution output\n', conv2d, '\nrun time in sec\n', run_time, '\nmean Square Error\n', mse)


if __name__ == '__main__':
    main()
