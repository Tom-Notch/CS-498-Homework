#!/usr/bin/env python3
"""
Question 5. Triangulation
In this question we move to 3D.
You are given keypoint matching between two images, together with the camera intrinsic and extrinsic matrix.
Your task is to perform triangulation to restore the 3D coordinates of the key points.
In your PDF, please visualize the 3d points and camera poses in 3D from three different viewing perspectives.
"""
import os
import random
from math import sqrt

import cv2  # our tested version is 4.5.5
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
all_good_matches = np.load("assets/all_good_matches.npy")

K1 = np.load("assets/fountain/Ks/0000.npy")
K2 = np.load("assets/fountain/Ks/0005.npy")

R1 = np.load("assets/fountain/Rs/0000.npy")
R2 = np.load("assets/fountain/Rs/0005.npy")

t1 = np.load("assets/fountain/ts/0000.npy")
t2 = np.load("assets/fountain/ts/0005.npy")


def triangulate(K1, K2, R1, R2, t1, t2, all_good_matches):
    """
    Arguments:
        K1: intrinsic matrix for image 1, dim: (3, 3)
        K2: intrinsic matrix for image 2, dim: (3, 3)
        R1: rotation matrix for image 1, dim: (3, 3)
        R2: rotation matrix for image 1, dim: (3, 3)
        t1: translation for image 1, dim: (3,)
        t2: translation for image 1, dim: (3,)
        all_good_matches:  dim: (#matches, 4)
    Returns:
        points_3d, dim: (#matches, 3)
    """
    points_3d = None
    # --------------------------- Begin your code here ---------------------------------------------
    n = all_good_matches.shape[0]

    points_3d = np.zeros((n, 3))

    P1 = np.concatenate((R1, t1), axis=-1)
    P2 = np.concatenate((R2, t2), axis=-1)

    x1 = all_good_matches[:, :2]
    x2 = all_good_matches[:, 2:]

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    x1_c = (K1_inv @ np.concatenate((x1, np.ones((n, 1))), axis=1).T).T
    x2_c = (K2_inv @ np.concatenate((x2, np.ones((n, 1))), axis=1).T).T

    for i in range(n):
        A = np.array(
            [
                x1_c[i, 1] * P1[2] - P1[1],
                P1[0] - x1_c[i, 0] * P1[2],
                x2_c[i, 1] * P2[2] - P2[1],
                P2[0] - x2_c[i, 0] * P2[2],
            ]
        )

        _, _, V = np.linalg.svd(A)

        points_3d[i] = (V[3] / V[3, 3])[0:3]
    # --------------------------- End your code here   ---------------------------------------------
    return points_3d


points_3d = triangulate(K1, K2, R1, R2, t1, t2, all_good_matches)
if points_3d is not None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Visualize both point and camera
    # Check this link for Open3D visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Check this function for adding a virtual camera in the visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Open3D is not the only option. You could use matplotlib, vtk or other visualization tools as well.
    # --------------------------- Begin your code here ---------------------------------------------
    W = 3072
    H = 2048
    cam_1 = o3d.geometry.LineSet.create_camera_visualization(
        W, H, K1, np.vstack((np.concatenate((R1, t1), axis=-1), [0, 0, 0, 1]))
    )
    cam_2 = o3d.geometry.LineSet.create_camera_visualization(
        W, H, K2, np.vstack((np.concatenate((R2, t2), axis=-1), [0, 0, 0, 1]))
    )
    o3d.visualization.draw_geometries([pcd, cam_1, cam_2])
    # --------------------------- End your code here   ---------------------------------------------
