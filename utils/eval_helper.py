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

import math
import cv2
import numpy as np

# Moved some functions to third_party to be safe
# Reference: https://github.com/Khrylx/Mujoco-modeler/blob/master/transformation.py
from third_party.utils.eval_helper import align, align_model, \
        quaternion_matrix, quaternion_from_matrix

_EPS = np.finfo(float).eps * 4.0


def calc_trans_error(model, data):
    alignment_error = model - data
    sqrt_val = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error),
                              0))
    return sqrt_val


def calc_num_iter_ransac(prob_inlier=0.5,
                         num_min_sample=3,
                         prob_success=0.9999):
    if prob_inlier == 1:
        return 1
    else:
        return math.log(1 - prob_success) / math.log(
            1 - math.pow(prob_inlier, num_min_sample))


def calc_std(xyz):
    xyz_zerocentered = xyz - xyz.mean(1, keepdims=True)
    xyz_std = np.sqrt((xyz_zerocentered * xyz_zerocentered).sum(0).mean())
    return xyz_std


def calc_max_trans_error(xyz):
    x = xyz[0, :].max() - xyz[0, :].min()
    y = xyz[1, :].max() - xyz[1, :].min()
    z = xyz[2, :].max() - xyz[2, :].min()
    trans_error_ali = math.sqrt(x * x / 4 + y * y / 4 + z * z / 4)
    return trans_error_ali


def ate_ransac(model, data, num_itr, threshold):

    org_trans_error = calc_trans_error(model, data).mean()
    opt_trans_error_inlier = org_trans_error
    opt_rot = np.identity(3)
    opt_trans = np.zeros((3, 1))
    opt_scale = 1
    opt_num_inlier = 0
    for i in range(num_itr):
        # draw random sample
        idx = np.arange(model.shape[1])
        np.random.shuffle(idx)
        idx = idx[0:3]

        model_sample = model[:, idx]
        data_sample = data[:, idx]
        # align on samples
        rot, trans, scale = align(model_sample, data_sample)
        model_aligned = align_model(model, rot, trans, scale)
        trans_error = calc_trans_error(model_aligned, data)
        num_inlier = np.asarray(np.where(trans_error < threshold)).shape[1]

        if num_inlier == 0:
            continue

        avg_trans_error_inlier = trans_error[np.where(
            trans_error < threshold)].squeeze().mean()
        # avg_trans_error = trans_error.mean()
        if (i == 0 or opt_num_inlier < num_inlier
                or (opt_num_inlier == num_inlier
                    and opt_trans_error_inlier > avg_trans_error_inlier)):
            opt_num_inlier = num_inlier
            # opt_trans_error = avg_trans_error
            opt_trans_error_inlier = avg_trans_error_inlier
            opt_rot = rot
            opt_trans = trans
            opt_scale = scale
    return opt_rot, opt_trans, opt_scale, opt_trans_error_inlier, \
        opt_num_inlier


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv2.recoverPose(E, p1n, p2n)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            err_q = np.pi
            err_t = np.pi / 2

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t
