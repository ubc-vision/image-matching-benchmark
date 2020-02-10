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

import cv2
import numpy as np

from utils.stereo_helper import normalize_keypoints
from skimage.measure import ransac as skransac
from skimage.transform import FundamentalMatrixTransform
from methods.geom_models.common import _fail, _preprocess


def _skimage_estimate_E_without_intrinsics(cfg, matches, kps1, kps2, calib1,
                                           calib2):
    '''Estimate the Essential matrix from correspondences. Computes the
    Fundamental Matrix first and then retrieves the Essential matrix assuming
    known intrinsics.
    '''

    # Reference: https://docs.opencv.org/3.4.7/d9/d0c/group__calib3d.html
    # Defalt values for confidence: 0.99 (F), 0.999 (E)
    # Default value for the reprojection threshold: 3
    # (We set them to -1 when not applicable as OpenCV complains otherwise)
    method_geom = cfg.method_geom['method']
    if method_geom.lower() == 'skimage-ransac-f':
        min_matches = 9
        cv_reprojection_threshold = cfg.method_geom['threshold']
        cv_confidence = cfg.method_geom['confidence']
        max_iters = cfg.method_geom['max_iter']
    else:
        raise ValueError('Unknown method to estimate F')

    is_valid, matches, kp1, kp2 = _preprocess(matches, kps1, kps2, min_matches)
    if not is_valid:
        return _fail()
    if len(kp1) < 9:
        return _fail()
    try:
        F, mask_F = skransac((kp1, kp2),
                             FundamentalMatrixTransform,
                             min_samples=8,
                             residual_threshold=cv_reprojection_threshold,
                             max_trials=max_iters,
                             stop_probability=cv_confidence)
    except Exception:
        return _fail()

    mask_F = mask_F.astype(bool).flatten()
    F = F.params
    # OpenCV can return multiple values as 6x3 or 9x3 matrices
    if F is None:
        return _fail()
    elif F.shape[0] != 3:
        Fs = np.split(F, len(F) / 3)
    else:
        Fs = [F]

    # Find the best F
    K1, K2 = calib1['K'], calib2['K']
    kp1n = normalize_keypoints(kp1, K1)
    kp2n = normalize_keypoints(kp2, K2)
    E, num_inlier = None, 0
    # mask_E_cheirality_check = None
    for _F in Fs:
        _E = np.matmul(np.matmul(K2.T, _F), K1)
        _E = _E.astype(np.float64)
        _num_inlier, _R, _t, _mask = cv2.recoverPose(_E, kp1n[mask_F],
                                                     kp2n[mask_F])
        if _num_inlier >= num_inlier:
            num_inlier = _num_inlier
            E = _E
            # This is unused for now
            # mask_E_cheirality_check = _mask.flatten().astype(bool)

    # Return the initial list of matches (from F)
    indices = matches[:, mask_F.flatten()]
    return E, indices


def estimate_essential(cfg,
                       matches,
                       kps1,
                       kps2,
                       calib1,
                       calib2,
                       scales1=None,
                       scales2=None,
                       ori1=None,
                       ori2=None,
                       img1_fname=None,
                       img2_fname=None,
                       A1=None,
                       A2=None,
                       descs1=None,
                       descs2=None):
    '''Estimate the Essential matrix from correspondences.

    Common entry point for all methods. Currently uses OpenCV to estimate E,
    with or without assuming known intrinsics.
    '''

    # Estimate E with OpenCV (assumes known intrinsics)
    method = cfg.method_geom['method'].lower()
    if method in ['skimage-ransac-f']:
        return _skimage_estimate_E_without_intrinsics(cfg, matches, kps1, kps2,
                                                      calib1, calib2)
    else:
        raise ValueError('Unknown method to estimate E/F')
