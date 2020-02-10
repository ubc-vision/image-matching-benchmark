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
import math
import torch
from utils.stereo_helper import normalize_keypoints
from third_party.DFE.dfe.models import NormalizedEightPointNet
from third_party.DFE.dfe.utils import compute_residual
import warnings


def _fail():
    '''Common return values in case of failure.'''
    return np.zeros(0), np.zeros((2, 0), dtype=np.int32)


def _preprocess(matches, kps1, kps2, min_num_matches):
    '''Common preprocessing of keypoints/matches.'''

    # Give up if we have no matches
    # Can happen with empty keypoint lists
    if matches.size == 0:
        return False, None, None, None

    kp1 = kps1[:, :2]
    kp2 = kps2[:, :2]

    kp1 = np.squeeze(kp1[matches[0]])
    kp2 = np.squeeze(kp2[matches[1]])
    if kp1.ndim == 1 or kp2.ndim == 1:
        return False, None, None, None

    if matches.shape[1] < min_num_matches:
        return False, None, None, None

    return True, matches, kp1, kp2


def _intel_estimate_E_without_intrinsics(cfg,
                                         matches,
                                         kps1,
                                         kps2,
                                         calib1,
                                         calib2,
                                         scales1=None,
                                         scales2=None,
                                         ori1=None,
                                         ori2=None,
                                         descs1=None,
                                         descs2=None):
    '''Estimate the Essential matrix from correspondences. Computes the
    Fundamental Matrix first and then retrieves the Essential matrix assuming
    known intrinsics.
    '''

    method_geom = cfg.method_geom['method']
    min_matches = 8
    threshold = cfg.method_geom['threshold']
    postprocess = cfg.method_geom['postprocess']
    is_valid, matches, kp1, kp2 = _preprocess(matches, kps1, kps2, min_matches)
    if not is_valid:
        return _fail()
    pts_dfe = np.concatenate([kp1.reshape(-1, 2), kp2.reshape(-1, 2)], axis=1)

    if (descs1 is not None) and (descs2 is not None):
        desc_distance = np.sqrt((
            (descs1[matches[0, :]] - descs2[matches[1, :]])**2).mean(axis=1) +
                                1e-9)
        desc_distance = desc_distance / (desc_distance.max() + 1e-6)
        desc_distance = desc_distance.reshape(-1, 1)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        prematch = matcher.knnMatch(descs1, descs2, k=2)
        matches1 = []
        ratio = []
        matches_set1 = set(matches[0, :])
        for m, n in prematch:
            if m.queryIdx in matches_set1:
                ratio.append(m.distance / n.distance)
        ratio = np.array(ratio).reshape(-1, 1)
        ratio = ratio / (ratio.max() + 1e-6)
        assert len(ratio) == len(desc_distance)

    else:
        warnings.warn('No descriptor provided for DFE', UserWarning)
        desc_distance = np.zeros((len(pts_dfe), 1))
    if (scales1 is not None) and (scales2 is not None):
        rel_scale = np.abs(scales1[matches[0, :]] -
                           scales2[matches[1, :]]).reshape(-1, 1)
        rel_scale = rel_scale / (1e-6 + rel_scale.max())
    else:
        warnings.warn('No scale provided for DFE', UserWarning)
        rel_scale = np.zeros((len(pts_dfe), 1))
    if (ori1 is not None) and (ori2 is not None):
        rel_orient = np.minimum(
            np.abs(ori1[matches[0, :]] - ori2[matches[1, :]]),
            np.abs(ori2[matches[1, :]] - ori1[matches[0, :]])).reshape(-1, 1)
        rel_orient = rel_orient / (1e-6 + rel_orient.max())
    else:
        warnings.warn('No orientation provided for DFE', UserWarning)
        rel_orient = np.zeros((len(pts_dfe), 1))
    side_info = np.concatenate([desc_distance, rel_scale, rel_orient, ratio],
                               axis=1)

    model = NormalizedEightPointNet(depth=3, side_info_size=4)

    # TODO re-upload weights
    model.load_state_dict(
        torch.load('DFE_phototourism120.pt', map_location=torch.device('cpu')))
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    except:
        device = torch.device('cpu')
    model = model.eval()
    model = model.to(device)
    pts_orig = pts_dfe.copy()
    pts_dfe = torch.from_numpy(pts_dfe).to(device).unsqueeze(0).float()
    side_info = torch.from_numpy(side_info).to(
        torch.float).to(device).unsqueeze(0)
    with torch.no_grad():
        F_est, rescaling_1, rescaling_2, weights = model(pts_dfe, side_info)
    F_est = rescaling_1.permute(0, 2, 1).bmm(F_est[-1].bmm(rescaling_2))
    F_est = F_est / F_est[:, -1, -1].unsqueeze(-1).unsqueeze(-1)
    F = F_est[0].data.cpu().numpy()
    mask_F = compute_residual(pts_orig, F) <= threshold
    mask_F = mask_F.astype(bool).flatten()
    score = mask_F.sum()
    F_best = F.T
    inliers_best = score
    if postprocess:
        import pyransac
        inliers_best = score
        for th in [25, 50, 75]:
            perc = np.percentile(weights, threshold)
            good = np.where(weights > perc)[0]
            if len(good) < 9:
                continue
            pts_ = pts_orig[good]
            #_F, _ = cv2.findFundamentalMat(pts_[:, 2:], pts_[:, :2], cv2.FM_LMEDS)
            _F, _mask_F = pyransac.findFundamentalMatrix(
                kp1[good],
                kp2[good],
                0.25,
                0.99999,
                500000,
                0,
                error_type='sampson',
                symmetric_error_check=True,
                enable_degeneracy_check=False)
            if _F is None:
                continue
            _mask_F = compute_residual(pts_orig, _F) <= threshold
            inliers = _mask_F.sum()
            if inliers > inliers_best:
                F_best = _F
                inliers_best = inliers
                mask_F = _mask_F

    if inliers_best < 8:
        return _fail()

    # OpenCV can return multiple values as 6x3 or 9x3 matrices
    F = F_best

    if F_best is None:
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
                       descs1=None,
                       descs2=None):
    '''Estimate the Essential matrix from correspondences.

    Common entry point for all methods. Currently uses cmp-pyransac to
    estimate E, with or without assuming known intrinsics.

    '''
    raise NotImplementedError(
        'This is a built-in port of "Deep fundamental matrix estimation", '
        'ECCV 2018. It does not work: temporarily disabled')

    method = cfg.method_geom['method'].lower()
    if method in ['intel-dfe-f']:
        with torch.no_grad():
            return _intel_estimate_E_without_intrinsics(
                cfg, matches, kps1, kps2, calib1, calib2, scales1, scales2,
                ori1, ori2, descs1, descs2)
    else:
        raise ValueError('Unknown method to estimate E/F')
