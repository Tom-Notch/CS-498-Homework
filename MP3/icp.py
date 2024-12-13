#!/usr/bin/env python3
import time
from re import I

import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def fit_rigid(src, tgt, point_to_plane=False):
    # Question 2: Rigid Transform Fitting
    # Implement this function
    # -------------------------
    tgt_norm = None

    if point_to_plane is True:
        tgt_norm = tgt[1]

    tgt = tgt[0]

    T = np.identity(4)

    if point_to_plane is True:
        N = src.shape[1]
        A = np.zeros((N, 6))
        b = np.zeros((N, 1))

        b = np.sum(
            np.multiply(tgt_norm.T, tgt.T) - np.multiply(tgt_norm.T, src.T), axis=1
        ).reshape(-1, 1)
        A[:, :3] = np.multiply(
            tgt_norm.T[:, [2, 0, 1]], src.T[:, [1, 2, 0]]
        ) - np.multiply(tgt_norm.T[:, [1, 2, 0]], src.T[:, [2, 0, 1]])
        A[:, 3:] = tgt_norm.T

        x = np.linalg.pinv(A) @ b

        T[:3, :3] = np.array(
            [[1, -x[2, 0], x[1, 0]], [x[2, 0], 1, -x[0, 0]], [-x[1, 0], x[0, 0], 1]]
        )
        T[:3, 3:] = x[3:, :]

    else:
        src_center = np.array([[np.mean(src[0])], [np.mean(src[1])], [np.mean(src[2])]])
        tgt_center = np.array([[np.mean(tgt[0])], [np.mean(tgt[1])], [np.mean(tgt[2])]])
        src_hat = src - src_center
        tgt_hat = tgt - tgt_center

        U, S, Vh = np.linalg.svd(tgt_hat @ src_hat.T)

        rect_S = np.eye(3)

        if np.linalg.det(U @ Vh) < 0:
            rect_S[2, 2] = -1

        R = U @ rect_S @ Vh

        t = tgt_center - R @ src_center

        T[0:3, 0:3] = R
        T[0:3, 3:] = t
        T[3, 3] = 1
    # -------------------------
    return T


# Question 4: deal with point_to_plane = True


def icp(source, target, init_pose=np.eye(4), max_iter=20, point_to_plane=False):
    src = np.asarray(source.points).T
    tgt = np.asarray(target.points).T

    # Question 3: ICP
    # Hint 1: using KDTree for fast nearest neighbour
    # Hint 3: you should be calling fit_rigid inside the loop
    # You implementation between the lines
    # ---------------------------------------------------
    if point_to_plane is True:
        target.normalize_normals()
        tgt_norm = np.asarray(target.normals).T

    T = init_pose
    transforms = []
    delta_Ts = []

    N_src = src.shape[1]
    N_tgt = tgt.shape[1]

    neighbor = KDTree(tgt.T)

    inlier_ratio = 0
    inlier_threshold = 0.06
    print("iter %d: inlier ratio: %.2f" % (0, inlier_ratio))

    for i in range(max_iter):
        T_delta = np.identity(4)

        previous_T_inv = np.linalg.inv(T)

        src_homogeneous = np.vstack((src, np.ones(N_src)))
        src_homogeneous = T @ src_homogeneous

        neighbor_distances, neighbor_indices = neighbor.query(src_homogeneous[0:3].T)
        inlier_ratio = (
            neighbor_distances < inlier_threshold
        ).sum() / neighbor_distances.size

        tgt_correspondences = np.take(tgt.T, neighbor_indices, axis=0).T.reshape(
            3, neighbor_indices.size
        )
        tgt_input = [tgt_correspondences]
        if point_to_plane is True:
            tgt_correspondences_norm = np.take(
                tgt_norm.T, neighbor_indices, axis=0
            ).T.reshape(3, neighbor_indices.size)
            tgt_input.append(tgt_correspondences_norm)

        T = fit_rigid(src, tgt_input, point_to_plane)

        T_delta = previous_T_inv @ T

        # ---------------------------------------------------

        if inlier_ratio > 0.999:
            break

        print("iter %d: inlier ratio: %.2f" % (i + 1, inlier_ratio))
        # relative update from each iteration
        delta_Ts.append(T_delta.copy())
        # pose estimation after each iteration
        transforms.append(T.copy())
    return transforms, delta_Ts


def rgbd2pts(color_im, depth_im, K):
    # Question 1: unproject rgbd to color point cloud, provide visualiation in your document
    # Your implementation between the lines
    # ---------------------------
    shape = depth_im.shape
    N = shape[0] * shape[1]
    color = np.zeros((N, 3))
    xyz = np.zeros((N, 3))

    color = color_im.reshape(N, 3).copy()

    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    im_coordinate = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), np.ones((N, 1))), axis=-1
    ).reshape((shape[0], shape[1], 3))

    K_inverse = np.linalg.inv(K)

    xyz = (K_inverse @ im_coordinate.reshape(N, 3).T).T * np.concatenate(
        (depth_im.reshape(N, 1), depth_im.reshape(N, 1), depth_im.reshape(N, 1)),
        axis=-1,
    )
    # ---------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


# TODO (Shenlong): please check that I set this question up correctly, it is called on line 136


def pose_error(estimated_pose, gt_pose):
    # Question 5: Translation and Rotation Error
    # Use equations 5-6 in https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
    # Your implementation between the lines
    # ---------------------------
    translation_error = np.linalg.norm(gt_pose[0:3, 3:] - estimated_pose[0:3, 3:])
    rotation_error = np.arccos(
        (np.trace(estimated_pose[0:3, 0:3] @ np.linalg.inv(gt_pose[0:3, 0:3])) - 1) / 2
    )
    error = {"t": translation_error, "R": rotation_error}
    # ---------------------------
    return error


def read_data(ind=0):
    K = np.loadtxt("data/camera-intrinsics.txt", delimiter=" ")
    depth_im = cv2.imread("data/frame-%06d.depth.png" % (ind), -1).astype(float)
    depth_im /= 1000.0  # depth is saved in 16-bit PNG in millimeters
    # set invalid depth to 0 (specific to 7-scenes dataset)
    depth_im[depth_im == 65.535] = 0
    T = np.loadtxt(
        "data/frame-%06d.pose.txt" % (ind)
    )  # 4x4 rigid transformation matrix
    color_im = cv2.imread("data/frame-%06d.color.jpg" % (ind), -1)
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB) / 255.0
    return color_im, depth_im, K, T


if __name__ == "__main__":

    # pairwise ICP

    # read color, image data and the ground-truth, converting to point cloud
    color_im, depth_im, K, T_tgt = read_data(0)
    target = rgbd2pts(color_im, depth_im, K)
    color_im, depth_im, K, T_src = read_data(40)
    source = rgbd2pts(color_im, depth_im, K)

    # downsampling and normal estimatoin
    source = source.voxel_down_sample(voxel_size=0.02)
    target = target.voxel_down_sample(voxel_size=0.02)
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # conduct ICP (your code)
    final_Ts, delta_Ts = icp(source, target, point_to_plane=True)

    # visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    ctr.set_front([-0.11651295252277051, -0.047982289143896774, -0.99202945108647766])
    ctr.set_lookat([0.023592929264511786, 0.051808635289583765, 1.7903649529102956])
    ctr.set_up([0.097655832648056065, -0.9860023571949631, -0.13513952033284915])
    ctr.set_zoom(0.42199999999999971)
    vis.add_geometry(source)
    vis.add_geometry(target)

    save_image = False

    # update source images
    for i in range(len(delta_Ts)):
        source.transform(delta_Ts[i])
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)

    # visualize camera
    h, w, c = color_im.shape
    tgt_cam = o3d.geometry.LineSet.create_camera_visualization(
        w, h, K, np.eye(4), scale=0.2
    )
    src_cam = o3d.geometry.LineSet.create_camera_visualization(
        w, h, K, np.linalg.inv(T_src) @ T_tgt, scale=0.2
    )
    pred_cam = o3d.geometry.LineSet.create_camera_visualization(
        w, h, K, np.linalg.inv(final_Ts[-1]), scale=0.2
    )

    gt_pose = np.linalg.inv(T_src) @ T_tgt
    pred_pose = np.linalg.inv(final_Ts[-1])
    p_error = pose_error(pred_pose, gt_pose)
    print("Ground truth pose:", gt_pose)
    print("Estimated pose:", pred_pose)
    print("Rotation/Translation Error", p_error)

    tgt_cam.paint_uniform_color((1, 0, 0))
    src_cam.paint_uniform_color((0, 1, 0))
    pred_cam.paint_uniform_color((0, 0.5, 0.5))
    vis.add_geometry(src_cam)
    vis.add_geometry(tgt_cam)
    vis.add_geometry(pred_cam)

    vis.run()
    vis.destroy_window()

    # Provide visualization of alignment with camera poses in write-up.
    # Print pred pose vs gt pose in write-up.
