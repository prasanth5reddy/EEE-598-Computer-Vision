import cv2
import numpy as np
import matplotlib.pyplot as plt


class HarrisCorners:
    def __init__(self, orig_img, window_size, k, threshold):
        self.orig_img = orig_img
        # convert image from color to grayscale
        self.img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        self.window_size = window_size
        self.k = k
        self.threshold = threshold

    def compute_gradients(self):
        #  Here OpenCV's Sobel filter is used to compute gradients
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1)
        return grad_x, grad_y

    @staticmethod
    def gradient_products(i_x, i_y):
        # compute gradient products
        i_xx = i_x * i_x
        i_xy = i_x * i_y
        i_yy = i_y * i_y
        return i_xx, i_xy, i_yy

    def sum_gradient_products(self, i_xx, i_xy, i_yy):
        # Here window function is used as Gaussian
        s_xx = cv2.GaussianBlur(i_xx, (self.window_size, self.window_size), 0)
        s_xy = cv2.GaussianBlur(i_xy, (self.window_size, self.window_size), 0)
        s_yy = cv2.GaussianBlur(i_yy, (self.window_size, self.window_size), 0)
        return s_xx, s_xy, s_yy

    def compute_response(self, s_xx, s_xy, s_yy):
        # compute determinant
        det_m = s_xx * s_yy - s_xy * s_xy
        # compute trace
        trace_m = s_xx + s_yy
        return det_m - self.k * trace_m * trace_m

    def threshold_corner(self, response):
        # set values to one if response is greater than certain threshold, else zero
        thresh_response = (np.abs(response) > self.threshold * np.max(response)) * 1
        # find indices where corner responses are non zero
        thresh_indices = np.nonzero(thresh_response)
        # store response indices are its value
        thresh_values = np.array([((i, j), response[i][j]) for i, j in zip(thresh_indices[0], thresh_indices[1])])
        return thresh_values

    @staticmethod
    def compute_nms(response, separation=5):
        # reverse sort the threshold response values
        sorted_response = sorted(response, key=lambda x: x[-1], reverse=True)
        # start nms indices list with index having maximum threshold value
        nms_indices = [sorted_response[0][0]]
        # loop through the next response indices to suppress non maximum values with in certain separation
        for ((response_i_x, response_i_y), _) in sorted_response:
            # if not found already maximum value with in separation, then insert into nms list
            if not np.sum([abs(response_i_x - nms_index[0]) < separation and abs(
                    response_i_y - nms_index[1]) < separation for nms_index in nms_indices]):
                nms_indices.append((response_i_x, response_i_y))
        return nms_indices

    def compute_corners(self):
        # 1. compute gradients of image in x and y direction
        i_x, i_y = self.compute_gradients()

        # show_image(i_x, color='gray')
        # show_image(i_y, color='gray')

        # 2. calculate product of the gradients
        i_xx, i_xy, i_yy = self.gradient_products(i_x, i_y)

        # 3. compute sum of the product of gradients
        s_xx, s_xy, s_yy = self.sum_gradient_products(i_xx, i_xy, i_yy)

        # 4. compute the response at each pixel using the matrix formed by [[s_xx, s_xy],
        #                                                                   [s_xy, s_yy]] at each pixel
        response = self.compute_response(s_xx, s_xy, s_yy)

        # show_image(response, color='gray')

        # 5. threshold the corner response
        # since threshold can be different for different images, maximum response * k is used as threshold
        thresh_response = self.threshold_corner(response)

        # 6. apply non-max suppression
        nms_indices = self.compute_nms(thresh_response, 5)
        return nms_indices

    def show_corners(self, img_title):
        # compute harris corners from scratch
        corners_indices = self.compute_corners()
        # create blank image to show corners
        corners_img = np.zeros_like(self.orig_img)
        # create a copy of original image
        img_with_corners = self.orig_img.copy()
        for index in corners_indices:
            # set corner points to 1 in grayscale image
            corners_img[index[0]][index[1]] = 1
            # make a circle with center on detected corners for visualisation
            cv2.circle(img_with_corners, (index[1], index[0]), 3, [0, 0, 255], -1)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        fig.suptitle(img_title)
        # show image with only corners
        axes[0].imshow(corners_img, cmap='gray')
        axes[0].set_title('Detected corners')
        # show original image with detected corners
        axes[1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Image with detected corners')
        # save original image with detected corners
        cv2.imwrite('data/detected_corners.jpg', img_with_corners)
        plt.show()
