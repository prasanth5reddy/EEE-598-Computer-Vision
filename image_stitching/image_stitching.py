import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def get_features(img):
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create sift feature detector
    sift = cv2.xfeatures2d.SIFT_create()
    # extract key points and descriptors for image
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def match_fd(kp_1, kp_2, des_1, des_2):
    bf_matcher = False
    if bf_matcher:
        # This cv2 matcher is used for rapidly testing the warped images
        # as euclidean distance computation is taking long time
        # In the submission this code shouldn't run
        matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        # get matches
        matches = matcher.match(des_1, des_2)
        # initialise matched image points list
        img_1_pts, img_2_pts = [], []
        for match in matches:
            # add matched points to the image point list
            pts_1 = list(kp_1[match.queryIdx].pt)
            pts_2 = list(kp_2[match.trainIdx].pt)
            img_1_pts.append(pts_1)
            img_2_pts.append(pts_2)
        return np.array(img_1_pts), np.array(img_2_pts)
    else:
        print('Computing matches between images.... (approx 2 minutes run time)')
        s = time.time()
        # set euclidean nearest neighbours ratio threshold
        threshold = 0.6
        #  initialise matched image points list
        img_1_pts, img_2_pts = [], []
        for i in range(des_1.shape[0]):
            # initialise the euclidean distance list
            euclid = []
            for j in range(des_2.shape[0]):
                # add euclid distance for each descriptor in image 2 and the corresponding keypoint
                euclid.append((np.linalg.norm(des_1[i] - des_2[j]), kp_2[j].pt))
            # sort the euclidean distances and find smallest two
            euclid = sorted(euclid, key=lambda x: x[0])[:2]
            # check ratio less than threshold
            if euclid[0][0] / euclid[1][0] < threshold:
                img_1_pts.append(list(kp_1[i].pt))
                img_2_pts.append(list(euclid[0][1]))
        print(f'Matches computed. Took {round(time.time() - s, 2)} seconds')
        return np.array(img_1_pts), np.array(img_2_pts)


def compute_H(img_1_pts, img_2_pts):
    # create empty list for matrix A
    matrix_a = []
    n = img_1_pts.shape[1]
    for i in range(n):
        # iterate through each pair of points
        # get the matching point pairs from each of the images
        (x_1, y_1), (x_2, y_2) = img_1_pts[:, i], img_2_pts[:, i]
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


def h_error(pt_1, pt_2, h):
    # reshape into vector
    pt_1 = np.append(pt_1, [1]).reshape(3, 1)
    # transform point 1 to homogenous space using h matrix
    pt_1_dash = np.matmul(h, pt_1)
    # scale the point such that last value is 1
    pt_1_dash = pt_1_dash / pt_1_dash[2][0]
    # compute euclidean distance between the transformed point 1 and point 2
    return np.linalg.norm(pt_1_dash[:2] - pt_2.reshape(2, 1))


def ransac(img_1_pts, img_2_pts, iterations, thresh):
    # initialize empty list of maximum inliers
    max_inliers = []
    for i in range(iterations):
        # find random four matching pair of points from both images
        random_size = 4
        random_indices = np.random.randint(0, img_1_pts.shape[0], random_size)
        pts_1, pts_2 = [], []
        for index in random_indices:
            pts_1.append(img_1_pts[index])
            pts_2.append(img_2_pts[index])
        pts_1, pts_2 = np.transpose(pts_1), np.transpose(pts_2)

        # compute homography between the above computed points
        h = compute_H(pts_1, pts_2)
        # initialize empty list of inliers satisfying the homography matrix
        inliers = []

        # estimate error between all matching points with obtained homography matrix
        error_thresh = 5
        for j in range(img_1_pts.shape[0]):
            if h_error(img_1_pts[j], img_2_pts[j], h) < error_thresh:
                # if error is minimal add it to inliers list
                inliers.append([img_1_pts[j], img_2_pts[j]])

        # if current model is better than previous model then update the maximum inliers
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
        # if a threshold of maximum inliers is achieved, then break out of the loop
        if len(max_inliers) > (img_1_pts.shape[0] * thresh):
            break
        # printing current and maximum inliers at each iteration
        print(f'Iteration : {i + 1}, max inliers : {len(max_inliers)}, current inliers : {len(inliers)}')
    return np.array(max_inliers)


def show_matches(img_1, img_2, matches, inliers=False):
    # combine the image
    img_comb = np.concatenate((img_1, img_2), axis=1)
    w = img_1.shape[1]
    # show the combines image
    plt.imshow(cv2.cvtColor(img_comb, cv2.COLOR_BGR2RGB), aspect='auto')
    # check if we are printing inliers or not
    if not inliers:
        plt.title('Matches')
        no_of_matches = matches.shape[0]
        color = 'r'
    else:
        plt.title('Inliers')
        no_of_matches = matches.shape[0]
        color = 'b'
    # take some random number of matches for better visualisation
    for i in np.random.randint(0, no_of_matches, 100):
        # for i in range(no_of_matches):
        x_1, y_1 = matches[i, 0, :]
        x_2, y_2 = matches[i, 1, :]
        # plot the line between matched points
        plt.plot(np.array([x_1, x_2 + w]), np.array([y_1, y_2]), color=color, linewidth=0.5)
    plt.imshow(cv2.cvtColor(img_comb, cv2.COLOR_BGR2RGB))
    plt.show()


def compute_final_homography(img_1, img_2):
    # find key points and descriptors
    kp_1, des_1 = get_features(img_1)
    kp_2, des_2 = get_features(img_2)
    # get matches
    img_1_pts, img_2_pts = match_fd(kp_1, kp_2, des_1, des_2)
    matches = np.array([[x, y] for x, y in zip(img_1_pts, img_2_pts)])
    # show matches
    show_matches(img_1, img_2, matches)
    print('matches shown')
    # run ransac
    iterations, threshold = 300, 0.5
    final_inliers = ransac(img_1_pts, img_2_pts, iterations, threshold)
    # show final inliers
    show_matches(img_1, img_2, final_inliers, inliers=True)
    print('inliers shown')
    # compute final homography
    final_h = compute_H(np.transpose(final_inliers[:, 0, :]), np.transpose(final_inliers[:, 1, :]))
    return final_h


def main():
    # read images
    img_a = cv2.imread('data/keble_a.jpg', 1)
    img_b = cv2.imread('data/keble_b.jpg', 1)
    img_c = cv2.imread('data/keble_c.jpg', 1)

    # compute final homography for image c and image b
    h_bc = compute_final_homography(img_c, img_b)
    # warp image c into reference frame of image b
    warped_c = cv2.warpPerspective(img_c, h_bc, (img_c.shape[1] + img_b.shape[1], img_c.shape[0]))
    plt.imshow(cv2.cvtColor(warped_c, cv2.COLOR_BGR2RGB))
    plt.title('Warped c')
    plt.show()
    # stitch image b and c
    stitched_ac = warped_c
    stitched_ac[:, :550] = img_b[:, :550]

    # compute final homography for image a and image b
    h_ab = compute_final_homography(img_a, img_b)
    # since translation along x axis is going negative, setting it to zero for easier stitching
    # can do it better using padding zeros around the image?
    h_ab[0][2] = 0
    # warp image c into reference frame of image b
    warped_a = cv2.warpPerspective(img_a, h_ab, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
    plt.imshow(cv2.cvtColor(warped_a, cv2.COLOR_BGR2RGB))
    plt.title('Warped a')
    plt.show()
    # stitch together all the images
    stitched_abc = warped_a
    stitched_abc[:, 300:] = stitched_ac[:, :stitched_abc.shape[1] - 300]
    # show final image
    plt.imshow(cv2.cvtColor(stitched_abc, cv2.COLOR_BGR2RGB))
    plt.title('Stitched Image')
    plt.show()
    # save stitched image
    cv2.imwrite('data/stitched_image.jpg', stitched_abc)


if __name__ == "__main__":
    main()
