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
from methods.geom_models.common import _fail, _preprocess


def _cv2_estimate_E_with_intrinsics(cfg, matches, kps1, kps2, calib1, calib2):
    '''Estimate the Essential matrix from correspondences. Assumes known
    intrinsics.
    '''

    # Reference: https://docs.opencv.org/3.4.7/d9/d0c/group__calib3d.html
    # Defalt values for confidence: 0.99 (F), 0.999 (E)
    # Default value for the reprojection threshold: 3
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']
    if geom['method'].lower() == 'cv2-ransac-e':
        cv_method = 'FM_RANSAC'
        cv_threshold = geom['threshold']
        cv_confidence = geom['confidence']
    elif geom['method'].lower() == 'cv2-lmeds-e':
        cv_method = 'FM_LMEDS'
        cv_threshold = None
        cv_confidence = geom['confidence']
    else:
        raise ValueError('Unknown method to estimate E')

    is_valid, matches, kp1, kp2 = _preprocess(matches, kps1, kps2, 5)
    if not is_valid:
        return _fail()

    # Normalize keypoints with ground truth intrinsics
    kp1_n = normalize_keypoints(kp1, calib1['K'])
    kp2_n = normalize_keypoints(kp2, calib2['K'])

    cv2.setRNGSeed(cfg.opencv_seed)
    E, mask_E = cv2.findEssentialMat(kp1_n,
                                     kp2_n,
                                     method=getattr(cv2, cv_method),
                                     threshold=cv_threshold,
                                     prob=cv_confidence)
    mask_E = mask_E.astype(bool).flatten()

    # OpenCV can return multiple values as 6x3 or 9x3 matrices
    if E is None:
        return _fail()
    elif E.shape[0] != 3:
        Es = np.split(E, len(E) / 3)
    # Or a single matrix
    else:
        Es = [E]

    # Find the best E
    E, num_inlier = None, 0
    # mask_E_cheirality_check = None
    for _E in Es:
        _num_inlier, _R, _t, _mask = cv2.recoverPose(_E, kp1_n[mask_E],
                                                     kp2_n[mask_E])
        if _num_inlier >= num_inlier:
            num_inlier = _num_inlier
            E = _E
            # This is unused for now
            # mask_E_cheirality_check = _mask.flatten().astype(bool)

    indices = matches[:, mask_E.flatten()]
    return E, indices


def _cv2_estimate_E_without_intrinsics(cfg, matches, kps1, kps2, calib1,
                                       calib2):
    '''Estimate the Essential matrix from correspondences. Computes the
    Fundamental Matrix first and then retrieves the Essential matrix assuming
    known intrinsics.
    '''

    # Reference: https://docs.opencv.org/3.4.7/d9/d0c/group__calib3d.html
    # Defalt values for confidence: 0.99 (F), 0.999 (E)
    # Default value for the reprojection threshold: 3
    # (We set them to -1 when not applicable as OpenCV complains otherwise)
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']
    if geom['method'].lower() in ['cv2-ransac-f', 'cv2-patched-ransac-f']:
        min_matches = 8
        cv_method = 'FM_RANSAC'
        cv_reprojection_threshold = geom['threshold']
        cv_confidence = geom['confidence']
        if geom['method'].lower() == 'cv2-patched-ransac-f':
            cv_max_iter = geom['max_iter']
    elif geom['method'].lower() == 'cv2-lmeds-f':
        min_matches = 8
        cv_method = 'FM_LMEDS'
        cv_reprojection_threshold = -1
        cv_confidence = geom['confidence']
    elif geom['method'].lower() == 'cv2-7pt':
        # This should actually be *equal* to 7? We'll probably never use it...
        min_matches = 7
        cv_method = 'FM_7POINT'
        cv_reprojection_threshold = -1
        cv_confidence = -1
    elif geom['method'].lower() == 'cv2-8pt':
        min_matches = 8
        cv_method = 'FM_8POINT'
        cv_reprojection_threshold = -1
        cv_confidence = -1
    else:
        raise ValueError('Unknown method to estimate F')

    is_valid, matches, kp1, kp2 = _preprocess(matches, kps1, kps2, min_matches)
    if not is_valid:
        return _fail()

    cv2.setRNGSeed(cfg.opencv_seed)

    # Temporary fix to allow for patched opencv
    if geom['method'].lower() == 'cv2-patched-ransac-f':
        F, mask_F = cv2.findFundamentalMat(
            kp1,
            kp2,
            method=getattr(cv2, cv_method),
            ransacReprojThreshold=cv_reprojection_threshold,
            confidence=cv_confidence,
            maxIters=cv_max_iter)
    else:
        F, mask_F = cv2.findFundamentalMat(
            kp1,
            kp2,
            method=getattr(cv2, cv_method),
            ransacReprojThreshold=cv_reprojection_threshold,
            confidence=cv_confidence)
    mask_F = mask_F.astype(bool).flatten()

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
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']
    
    method = geom['method'].lower()
    if method in ['cv2-ransac-e', 'cv2-lmeds-e']:
        return _cv2_estimate_E_with_intrinsics(cfg, matches, kps1, kps2,
                                               calib1, calib2)
    elif method in [
            'cv2-ransac-f', 'cv2-patched-ransac-f', 'cv2-lmeds-f', 'cv2-7pt',
            'cv2-8pt'
    ]:
        return _cv2_estimate_E_without_intrinsics(cfg, matches, kps1, kps2,
                                                  calib1, calib2)
    else:
        raise ValueError('Unknown method to estimate E/F')
