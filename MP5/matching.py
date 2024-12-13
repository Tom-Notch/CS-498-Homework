#!/usr/bin/env python3
import numpy as np
from matching_utils import iou


def get_index(cost):
    if len(cost) == 1:
        if cost == 0:
            return np.array([0, 0], dtype=int).reshape(1, -1), True
        else:
            return [], False

    for i in np.argwhere(cost[:, 0] == 0):
        rows = list(range(cost.shape[0]))
        rows.remove(i)
        next_cost = cost[:, 1:].take(rows, axis=0)
        matches, success = get_index(next_cost)
        if success:
            matches[:, -1] += 1
            matches[np.argwhere(matches[:, 0] >= i), 0] += 1
            matches = np.append(matches, [[int(i), 0]], axis=0)
            return matches[matches[:, 0].argsort()], True

    return [], False


def get_rows_columns(cost):
    tmp_cost = cost.copy()
    marked_rows = []
    marked_columns = []

    while np.min(tmp_cost) == 0:
        column_zeros = tmp_cost.shape[0] - np.count_nonzero(tmp_cost, axis=0)
        row_zeros = tmp_cost.shape[1] - np.count_nonzero(tmp_cost, axis=1)

        column_vs_row_zeros = np.array(np.meshgrid(column_zeros, row_zeros)).transpose(
            [1, 2, 0]
        )
        column_plus_row_zeros = np.sum(column_vs_row_zeros, axis=-1) - (tmp_cost == 0)

        pivot_point = np.unravel_index(
            np.argmax(column_plus_row_zeros), column_plus_row_zeros.shape
        )

        if np.min(tmp_cost[pivot_point[0]]) == 0:
            tmp_cost[pivot_point[0]] = 1
            marked_rows.append(pivot_point[0])

        if np.min(tmp_cost[:, pivot_point[1]]) == 0:
            tmp_cost[:, pivot_point[1]] = 1
            marked_columns.append(pivot_point[1])

    return np.sort(np.array(marked_rows)), np.sort(np.array(marked_columns))


def data_association(dets, trks, threshold=-0.2, algm="greedy"):
    """
    Q1. Assigns detections to tracked object

    dets:       a list of Box3D object
    trks:       a list of Box3D object
    threshold:  only mark a det-trk pair as a match if their iou distance is less than the threshold
    algm:       for extra credit, implement the hungarian algorithm as well

    Returns 3 lists:
        matches, kx2 np array of match indices, i.e. [[det_idx1, trk_idx1], [det_idx2, trk_idx2], ...]
        unmatched_dets, a 1d array of indices of unmatched detections
        unmatched_trks, a 1d array of indices of unmatched trackers
    """
    # Hint: you should use the provided iou(box_a, box_b) function to compute distance/cost between pairs of box3d objects
    # iou() is an implementation of a 3D box IoU

    # --------------------------- Begin your code here ---------------------------------------------
    if algm == "greedy":
        matches = []
        unmatched_dets = np.ones(len(dets))
        unmatched_trks = np.ones(len(trks))

        gious = np.zeros((len(dets), len(trks)))
        for i in range(len(dets)):
            for j in range(len(trks)):
                gious[i, j] = iou(dets[i], trks[j])

        while len(dets) > 0 and len(trks) > 0 and np.max(gious) >= threshold:
            idx = np.unravel_index(np.argmax(gious), gious.shape)
            unmatched_dets[idx[0]] = 0  # mask as selected
            unmatched_trks[idx[1]] = 0  # mask as selected
            matches.append([idx[0], idx[1]])

            gious[idx[0], :] = -1.0  # mask all selected matches
            gious[:, idx[1]] = -1.0

        unmatched_dets = np.array(*np.where(unmatched_dets > 0)).astype(int)
        unmatched_trks = np.array(*np.where(unmatched_trks > 0)).astype(int)

        matches = np.array(matches)

    else:
        if len(trks) == 0:  # initial situation
            return (
                np.array([]),
                np.arange(len(dets)).astype(int),
                np.array([]).astype(int),
            )

        matches = []

        # size of the gious and cost matrix, might need padding
        mat_size = max(len(dets), len(trks))
        # to avoid strange issue when even number of 0 occur when finding minimum cover lines
        gious = np.random.uniform(-10, -1, size=(mat_size, mat_size))
        for i in range(len(dets)):
            for j in range(len(trks)):
                current_iou = iou(dets[i], trks[j])
                if current_iou >= threshold:
                    gious[i, j] = current_iou

        cost = 1 - gious

        # Step 1
        cost = cost.T
        cost -= cost.min(axis=1).reshape(-1, 1)
        cost = cost.T

        tmp_matches, success = get_index(cost)

        if not success:
            # Step 2
            cost -= cost.min(axis=1).reshape(-1, 1)

            tmp_matches, success = get_index(cost)

        while not success:
            # Step 3
            marked_rows, marked_columns = get_rows_columns(cost)

            # Step 4
            unmarked_rows = np.ones(cost.shape[0])
            unmarked_rows[marked_rows] = 0
            unmarked_rows = np.argwhere(unmarked_rows).reshape(-1, 1)  # strange hack

            unmarked_columns = np.ones(cost.shape[0])
            unmarked_columns[marked_columns] = 0
            unmarked_columns = np.argwhere(unmarked_columns).reshape(-1)

            unmarked_chunks = cost[unmarked_rows, unmarked_columns]

            unmarked_chunks_min = np.min(unmarked_chunks)
            cost[unmarked_rows, unmarked_columns] -= unmarked_chunks_min
            cost[marked_rows.reshape(-1, 1), marked_columns] += unmarked_chunks_min

            # Step 5
            tmp_matches, success = get_index(cost)

        for i in range(len(tmp_matches)):
            index = tuple(tmp_matches[i])
            if gious[index] >= threshold:
                matches.append(index)
        matches = np.array(matches)

        unmatched_dets = np.ones(mat_size)
        unmatched_dets[matches[:, 0]] = 0
        unmatched_dets = np.argwhere(unmatched_dets).reshape(-1)
        unmatched_dets = unmatched_dets[unmatched_dets < len(dets)]
        unmatched_dets = np.array([*unmatched_dets]).astype(int)

        unmatched_trks = np.ones(mat_size)
        unmatched_trks[matches[:, 1]] = 0
        unmatched_trks = np.argwhere(unmatched_trks).reshape(-1)
        unmatched_trks = unmatched_trks[unmatched_trks < len(trks)]
        unmatched_trks = np.array([*unmatched_trks]).astype(int)
    # --------------------------- End your code here   ---------------------------------------------

    return matches, unmatched_dets, unmatched_trks
