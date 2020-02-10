# Copyright 2020 Google LLC, University of Victoria, Czech Technical University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from scipy.spatial import distance

from utils.eval_helper import eval_essential_matrix
from utils.load_helper import load_depth
from utils.path_helper import (get_stereo_pose_file, get_pairs_per_threshold,
                               get_data_path, get_geom_file)


def is_stereo_complete(cfg):
    '''Checks if stereo evaluation is complete.'''

    # Load pre-computed pairs with the new visibility criteria
    data_dir = get_data_path(cfg)
    pairs_per_th = get_pairs_per_threshold(data_dir)

    # Check if all files exist
    files = []
    for th in [None] + list(pairs_per_th.keys()):
        files += [get_stereo_pose_file(cfg, th)]

    any_missing = False
    for f in files:
        if not os.path.exists(f):
            any_missing = True
            break

    return not any_missing


def is_geom_complete(cfg):
    '''Checks if match computation is complete.'''

    is_complete = os.path.exists(get_geom_file(cfg))

    return is_complete


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def unnormalize_keypoints(keypoints, K):
    '''Undo the normalization of the keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = keypoints * np.array([[f_x, f_y]]) + np.array([[C_x, C_y]])

    return keypoints


def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero,
        -v[:, 2],
        v[:, 1],
        v[:, 2],
        zero,
        -v[:, 0],
        -v[:, 1],
        v[:, 0],
        zero,
    ],
                 axis=1)

    return M


def get_episym(x1n, x2n, dR, dt):

    # Fix crash when passing a single match
    if x1n.ndim == 1:
        x1n = x1n[None, ...]
        x2n = x2n[None, ...]

    num_pts = len(x1n)

    # Make homogeneous coordinates
    x1n = np.concatenate([x1n, np.ones((num_pts, 1))], axis=-1).reshape(
        (-1, 3, 1))
    x2n = np.concatenate([x2n, np.ones((num_pts, 1))], axis=-1).reshape(
        (-1, 3, 1))

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
                            dR).reshape(-1, 3, 3),
                  num_pts,
                  axis=0)

    x2Fx1 = np.matmul(x2n.transpose(0, 2, 1), np.matmul(F, x1n)).flatten()
    Fx1 = np.matmul(F, x1n).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2n).reshape(-1, 3)

    ys = x2Fx1**2 * (1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) + 1.0 /
                     (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()


def get_truesym(x1, x2, x1p, x2p):
    if len(x1) == 0 or len(x1p) == 0:
        return []

    ys1 = np.sqrt(np.sum((x1p - x2) * (x1p - x2), axis=1))
    ys2 = np.sqrt(np.sum((x2p - x1) * (x2p - x1), axis=1))
    ys = (ys1 + ys2) / 2
    ys = ys2

    return ys.flatten()


def get_projected_kp(x1, x2, d1, d2, dR, dT):
    # Append depth to key points
    y1 = np.concatenate([x1 * d1, d1], axis=1)
    y2 = np.concatenate([x2 * d2, d2], axis=1)

    # Project points from one image to another image
    y1p = np.matmul(dR[None], y1[..., None]) + dT[None]
    y2p = np.matmul(np.transpose(dR)[None], y2[..., None]) - \
        np.matmul(np.transpose(dR), dT)[None]

    # Move back to canonical plane
    x1p = np.squeeze(y1p[:, 0:2] / y1p[:, [2]])
    x2p = np.squeeze(y2p[:, 0:2] / y2p[:, [2]])

    return x1p, x2p


def get_repeatability(kp1n_p, kp2n, th_list):
    if kp1n_p.shape[0] == 0:
        return [0] * len(th_list)

    # Construct distance matrix
    # dis_mat = (np.tile(np.dot(kp1n_p * kp1n_p, np.ones([2, 1])),
    #                    (1, kp2n.shape[0])) +
    #            np.tile(np.transpose(np.dot(kp2n * kp2n, np.ones([2, 1]))),
    #                    (kp1n_p.shape[0], 1)) -
    #            2 * np.dot(kp1n_p, np.transpose(kp2n)))

    # Eduard: Extremely slow, this should be better
    dis_mat = distance.cdist(kp1n_p, kp2n, metric='sqeuclidean')

    # Get min for each point in kp1n_p
    min_array = np.amin(dis_mat, 1)

    # Calculate repeatability
    rep_score_list = []
    for th in th_list:
        rep_score_list.append((min_array < th * th).sum() / kp1n_p.shape[0])

    return rep_score_list


def eval_match_score(kp1, kp2, kp1n, kp2n, kp1p, kp2p, d1, d2, inl, dR, dT):
    # Fail silently if there are no inliers
    if inl.size == 0:
        return np.array([]), np.array([])

    # Fix crash when this is a single element
    # Should not happen but it seems that it does?
    if inl.ndim == 1:
        inl = inl[..., None]

    kp1_inl = kp1[inl[0]]
    kp2_inl = kp2[inl[1]]
    kp1p_inl = kp1p[inl[0]]
    kp2p_inl = kp2p[inl[1]]
    kp1n_inl = kp1n[inl[0]]
    kp2n_inl = kp2n[inl[1]]
    d1_inl = d1[inl[0]]
    d2_inl = d2[inl[1]]

    nonzero_index = np.nonzero(np.squeeze(d1_inl * d2_inl))

    # Get the geodesic distance in normalized coordinates
    geod_d = get_episym(kp1n_inl, kp2n_inl, dR, dT)

    # Get the projected distance in image coordinates
    true_d = get_truesym(kp1_inl[nonzero_index], kp2_inl[nonzero_index],
                         kp1p_inl[nonzero_index], kp2p_inl[nonzero_index])

    return geod_d, true_d


def compute_stereo_metrics_from_E(img1, img2, depth1, depth2, kp1, kp2, calib1,
                                  calib2, E, inl_prematch, inl_refined,
                                  inl_geom, cfg):
    ''' Computes the stereo metrics.'''

    # t = time()
    # Clip keypoints based on shape of matches
    kp1 = kp1[:, :2]
    kp2 = kp2[:, :2]

    # Load depth map
    dm1 = load_depth(depth1)
    dm2 = load_depth(depth2)
    img1_shp = dm1.shape
    img2_shp = dm2.shape

    # Get depth for each keypoint
    kp1_int = np.round(kp1).astype(int)
    kp2_int = np.round(kp2).astype(int)

    # Some methods can give out-of-bounds keypoints close to image boundaries
    # Safely marked them as occluded
    valid1 = (kp1_int[:, 0] >= 0) & (kp1_int[:, 0] < dm1.shape[1]) & (
        kp1_int[:, 1] >= 0) & (kp1_int[:, 1] < dm1.shape[0])
    valid2 = (kp2_int[:, 0] >= 0) & (kp2_int[:, 0] < dm2.shape[1]) & (
        kp2_int[:, 1] >= 0) & (kp2_int[:, 1] < dm2.shape[0])
    d1 = np.zeros((kp1_int.shape[0], 1))
    d2 = np.zeros((kp2_int.shape[0], 1))
    d1[valid1, 0] = dm1[kp1_int[valid1, 1], kp1_int[valid1, 0]]
    d2[valid2, 0] = dm2[kp2_int[valid2, 1], kp2_int[valid2, 0]]

    # Get R, t from calibration information
    R_1, t_1 = calib1['R'], calib1['T'].reshape((3, 1))
    R_2, t_2 = calib2['R'], calib2['T'].reshape((3, 1))

    # Compute dR, dt
    dR = np.dot(R_2, R_1.T)
    dT = t_2 - np.dot(dR, t_1)

    # Project the keypoints using depth
    kp1n = normalize_keypoints(kp1, calib1['K'])
    kp2n = normalize_keypoints(kp2, calib2['K'])
    kp1n_p, kp2n_p = get_projected_kp(kp1n, kp2n, d1, d2, dR, dT)
    kp1_p = unnormalize_keypoints(kp1n_p, calib2['K'])
    kp2_p = unnormalize_keypoints(kp2n_p, calib1['K'])

    # Get non zero depth index
    d1_nonzero_idx = np.nonzero(np.squeeze(d1))
    d2_nonzero_idx = np.nonzero(np.squeeze(d2))

    # Get index of projected kp inside image
    kp1_p_valid_idx = np.where((kp1_p[:, 0] < img2_shp[1])
                               & (kp1_p[:, 1] < img2_shp[0]))
    kp2_p_valid_idx = np.where((kp2_p[:, 0] < img1_shp[1])
                               & (kp2_p[:, 1] < img1_shp[0]))
    # print('Preprocessing: {}'.format(time() - t))

    # Calculate repeatability
    # Thresholds are hardcoded
    rep_s_list_1 = get_repeatability(
        kp1_p[np.intersect1d(kp1_p_valid_idx, d1_nonzero_idx)], kp2,
        cfg.matching_score_and_repeatability_px_threshold)
    rep_s_list_2 = get_repeatability(
        kp2_p[np.intersect1d(kp2_p_valid_idx, d2_nonzero_idx)], kp1,
        cfg.matching_score_and_repeatability_px_threshold)
    rep_s_list = [(rep_s_1 + rep_s_2) / 2
                  for rep_s_1, rep_s_2 in zip(rep_s_list_1, rep_s_list_2)]

    # Evaluate matching score after initial matching
    geod_d_list = []
    true_d_list = []
    geod_d, true_d = eval_match_score(kp1, kp2, kp1n, kp2n, kp1_p, kp2_p, d1,
                                      d2, inl_prematch, dR, dT)
    geod_d_list.append(geod_d)
    true_d_list.append(true_d)

    # Evaluate matching score after inlier refinement
    if inl_refined is None:
        geod_d_list.append([])
        true_d_list.append([])
    else:
        geod_d, true_d = eval_match_score(kp1, kp2, kp1n, kp2n, kp1_p, kp2_p,
                                          d1, d2, inl_refined, dR, dT)
        geod_d_list.append(geod_d)
        true_d_list.append(true_d)

    # Evaluate matching score after final geom
    geod_d, true_d = eval_match_score(kp1, kp2, kp1n, kp2n, kp1_p, kp2_p, d1,
                                      d2, inl_geom, dR, dT)
    geod_d_list.append(geod_d)
    true_d_list.append(true_d)

    # Compute error in R, T
    err_q, err_t = eval_essential_matrix(kp1n[inl_geom[0]], kp2n[inl_geom[1]],
                                         E, dR, dT)

    return geod_d_list, true_d_list, err_q, err_t, rep_s_list, True
