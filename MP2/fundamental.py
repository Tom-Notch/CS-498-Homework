#!/usr/bin/env python3
"""
Questions 2-4. Fundamental matrix estimation

Question 2. Eight-point Estimation
For this question, your task is to implement normalized and unnormalized eight-point algorithms to find out the fundamental matrix between two cameras.
We've provided a method to compute the average geometric distance, which is the distance between each projected keypoint from one image to its corresponding epipolar line in the other image.
You might consider reading that code below as a reminder for how we can use the fundamental matrix.
For more information on the normalized eight-point algorithm, please see this link: https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm

Question 3. RANSAC
Your task is to implement RANSAC to find out the fundamental matrix between two cameras if the correspondences are noisy.

Please report the average geometric distance based on your estimated fundamental matrix, given 1, 100, and 10000 iterations of RANSAC.
Please also visualize the inliers with your best estimated fundamental matrix in your solution for both images (we provide a visualization function).
In your PDF, please also explain why we do not perform SVD or do a least-square over all the matched key points.

Question 4. Visualizing Epipolar Lines
Please visualize the epipolar line for both images for your estimated F in Q2 and Q3.

To draw it on images, cv2.line, cv2.circle are useful to plot lines and circles.
Check our Lecture 4, Epipolar Geometry, to learn more about equation of epipolar line.
Our Lecture 4 and 5 cover most of the concepts here.
This link also gives a thorough review of epipolar geometry:
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
"""
import os
import random
from math import sqrt
from pathlib import Path

import cv2  # our tested version is 4.5.5
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

basedir = Path("assets/fountain")
img1 = cv2.imread(str(basedir / "images/0000.png"), 0)
img2 = cv2.imread(str(basedir / "images/0005.png"), 0)

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img1, cmap="gray")
axarr[1].imshow(img2, cmap="gray")
plt.show()

# --------------------- Question 2


def calculate_geometric_distance(all_matches, F):
    """
    Calculate average geomtric distance from each projected keypoint from one image to its corresponding epipolar line in another image.
    Note that you should take the average of the geometric distance in two direction (image 1 to 2, and image 2 to 1)
    Arguments:
        all_matches: all matched keypoint pairs that loaded from disk (#all_matches, 4).
        F: estimated fundamental matrix, (3, 3)
    Returns:
        average geomtric distance.
    """
    ones = np.ones((all_matches.shape[0], 1))
    all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Final distance.
    dist1 = d1.sum() / all_matches.shape[0]
    dist2 = d2.sum() / all_matches.shape[0]

    dist = (dist1 + dist2) / 2
    return dist


# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
eight_good_matches = np.load("assets/eight_good_matches.npy")
all_good_matches = np.load("assets/all_good_matches.npy")


def estimate_fundamental_matrix(matches, normalize=False):
    """
    Arguments:
        matches: Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    F = np.eye(3)
    # --------------------------- Begin your code here ---------------------------------------------
    matches = matches.copy()
    n = matches.shape[0]
    d = 9
    A = np.zeros((n, d))

    if normalize is True:
        centroids = np.array([np.mean(matches[:, i]) for i in range(matches.shape[1])])
        for i in range(matches.shape[1]):
            matches[:, i] -= centroids[i]
        scaling_factors = np.array(
            [
                np.sqrt(2) / np.mean(np.hypot(matches[:, 0], matches[:, 1])),
                np.sqrt(2) / np.mean(np.hypot(matches[:, 2], matches[:, 3])),
            ]
        )
        matches[:, 0:2] *= scaling_factors[0]
        matches[:, 2:4] *= scaling_factors[1]
        T_prime = np.array(
            [
                [scaling_factors[0], 0.0, -centroids[0] * scaling_factors[0]],
                [0.0, scaling_factors[0], -centroids[1] * scaling_factors[0]],
                [0.0, 0.0, 1.0],
            ]
        )
        T = np.array(
            [
                [scaling_factors[1], 0.0, -centroids[2] * scaling_factors[1]],
                [0.0, scaling_factors[1], -centroids[3] * scaling_factors[1]],
                [0.0, 0.0, 1.0],
            ]
        )

    for i in range(n):
        A[i] = [
            matches[i, 0] * matches[i, 2],
            matches[i, 0] * matches[i, 3],
            matches[i, 0],
            matches[i, 1] * matches[i, 2],
            matches[i, 1] * matches[i, 3],
            matches[i, 1],
            matches[i, 2],
            matches[i, 3],
            1,
        ]

    _, _, V = np.linalg.svd(A)

    F = V[d - 1].reshape((3, 3))

    U, S, V = np.linalg.svd(F)

    S[2] = 0

    F = U @ np.diag(S) @ V

    if normalize is True:
        F = T_prime.T @ F @ T
    # --------------------------- End your code here   ---------------------------------------------
    return F


F_with_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=True)
F_without_normalization = estimate_fundamental_matrix(
    eight_good_matches, normalize=False
)

# Evaluation (these numbers should be quite small)
print(
    f"F_with_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_with_normalization)}"
)
print(
    f"F_without_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_without_normalization)}"
)


# --------------------- Question 3


def ransac(all_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold):
    """
    Arguments:
        all_matches: coords of matched keypoint pairs in image 1 and 2, dims (# matches, 4).
        num_iteration: total number of RANSAC iteration
        estimate_fundamental_matrix: your eight-point algorithm function but use normalized version
        inlier_threshold: threshold to decide if one point is inlier
    Returns:
        best_F: best Fundamental matrix, dims (3, 3).
        inlier_matches_with_best_F: (#inliers, 4)
        avg_geo_dis_with_best_F: float
    """

    best_F = np.eye(3)
    inlier_matches_with_best_F = None
    avg_geo_dis_with_best_F = 0.0

    ite = 0
    # --------------------------- Begin your code here ---------------------------------------------
    max_num_inlier = 0

    while ite < num_iteration:
        ite += 1

        current_num_inlier = 0
        current_inlier_list = []

        # random sample correspondences
        random_sample = all_matches[
            np.random.choice(all_matches.shape[0], size=8, replace=False)
        ]

        # estimate the minimal fundamental estimation problem
        F = estimate_fundamental_matrix(random_sample, normalize=True)

        # compute # of inliers
        for i in range(random_sample.shape[0]):
            if (
                calculate_geometric_distance(random_sample[i].reshape(1, 4), F)
                < inlier_threshold
            ):
                current_num_inlier += 1
                current_inlier_list.append(random_sample[i])

        # update the current best solution
        if current_num_inlier > max_num_inlier or (
            current_num_inlier == max_num_inlier
            and calculate_geometric_distance(all_matches, F)
            < calculate_geometric_distance(all_matches, best_F)
        ):
            max_num_inlier = current_num_inlier
            best_F = F.copy()
            inlier_matches_with_best_F = (
                np.array(current_inlier_list).copy().reshape(max_num_inlier, 4)
            )
            avg_geo_dis_with_best_F = calculate_geometric_distance(all_matches, F)
    # --------------------------- End your code here   ---------------------------------------------
    return best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F


def visualize_inliers(im1, im2, inlier_coords):
    for i, im in enumerate([im1, im2]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(im, cmap="gray")
        plt.scatter(
            inlier_coords[:, 2 * i],
            inlier_coords[:, 2 * i + 1],
            marker="x",
            color="red",
            s=10,
        )
    plt.show()


num_iterations = [1, 100, 10000]
inlier_threshold = 0.17  # TODO: change the inlier threshold by yourself
for num_iteration in num_iterations:
    best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(
        all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold
    )
    if inlier_matches_with_best_F is not None:
        print(
            f"num_iterations: {num_iteration}; avg_geo_dis_with_best_F: {avg_geo_dis_with_best_F};"
        )
        visualize_inliers(img1, img2, inlier_matches_with_best_F)

# --------------------- Question 4


def visualize(estimated_F, img1, img2, kp1, kp2):
    # --------------------------- Begin your code here ---------------------------------------------
    img1 = img1.copy()
    img2 = img2.copy()
    for i in range(kp1.shape[0]):
        cv2.circle(img1, (int(kp1[i, 0]), int(kp1[i, 1])), 5, (255, 255, 255), -1)
        l_prime = (estimated_F @ np.array([kp2[i, 0], kp2[i, 1], 1.0])).reshape(-1)
        y_prime_zero = -l_prime[2] / l_prime[1]
        x_prime_max = img1.shape[1] - 1
        y_prime_max = -(x_prime_max * l_prime[0] + l_prime[2]) / l_prime[1]
        cv2.line(
            img1,
            (0, int(y_prime_zero)),
            (int(x_prime_max), int(y_prime_max)),
            (255, 255, 255),
        )

        cv2.circle(img2, (int(kp2[i, 0]), int(kp2[i, 1])), 5, (255, 255, 255), -1)
        l = (np.array([kp1[i, 0], kp1[i, 1], 1.0]) @ estimated_F).reshape(-1)
        y_zero = -l[2] / l[1]
        x_max = img2.shape[1] - 1
        y_max = -(x_max * l[0] + l[2]) / l[1]
        cv2.line(img2, (0, int(y_zero)), (int(x_max), int(y_max)), (255, 255, 255))

    plt.subplot(121)
    plt.imshow(img2, cmap="gray")
    plt.subplot(122)
    plt.imshow(img1, cmap="gray")

    plt.show()
    # --------------------------- End your code here   ---------------------------------------------


all_good_matches = np.load("assets/all_good_matches.npy")
all_good_matches = np.load("assets/all_good_matches.npy")
F_Q2 = F_with_normalization  # link to your estimated F in Q2
F_Q3_1, inliers_1, avg_geo_dis_1 = ransac(
    all_good_matches,
    num_iteration=1,
    estimate_fundamental_matrix=estimate_fundamental_matrix,
    inlier_threshold=0.17,
)  # link to your estimated F in Q3
F_Q3_100, inliers_100, avg_geo_dis_100 = ransac(
    all_good_matches,
    num_iteration=100,
    estimate_fundamental_matrix=estimate_fundamental_matrix,
    inlier_threshold=0.17,
)
F_Q3_10000, inliers_10000, avg_geo_dis_10000 = ransac(
    all_good_matches,
    num_iteration=10000,
    estimate_fundamental_matrix=estimate_fundamental_matrix,
    inlier_threshold=0.17,
)
visualize(F_Q2, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
print(calculate_geometric_distance(all_good_matches, F_Q2))
visualize(F_Q3_1, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
print(inliers_1.shape[0], avg_geo_dis_1)
visualize(F_Q3_100, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
print(inliers_100.shape[0], avg_geo_dis_100)
visualize(F_Q3_10000, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
print(inliers_10000.shape[0], avg_geo_dis_10000)
