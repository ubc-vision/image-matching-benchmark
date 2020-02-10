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

try:
    import pyransac
except Exception:
    print('WARNING: Could not import pyransac')
    pass
try:
    import pygcransac
except Exception:
    print('WARNING: Could not import pygcransac')
    pass
try:
    import pymagsac
except Exception:
    print('WARNING: Could not import pymagsac')
    pass

from utils.stereo_helper import normalize_keypoints
from methods.geom_models.common import _fail, _preprocess


def get_LAF(kps, sc, ori, A=None):
    '''
    Converts OpenCV keypoints into transformation matrix
    and pyramid index to extract from for the patch extraction
    '''
    num = len(kps)
    out = np.zeros((num, 6)).astype(np.float64)
    if A is not None:
        for i, kp in enumerate(kps):
            out[i, :2] = kps[i]
            s = 12 * sc[i]
            a = ori[i]
            cos = math.cos(a * math.pi / 180.0)
            sin = math.sin(a * math.pi / 180.0)
            Arot = np.zeros((2, 2))
            Arot[0, 0] = s * cos
            Arot[0, 1] = s * sin
            Arot[1, 0] = -s * sin
            Arot[1, 1] = s * cos
            Afin = np.matmul(Arot, A[i])
            out[i, 2] = Afin[0, 0]
            out[i, 3] = Afin[0, 1]
            out[i, 4] = Afin[1, 0]
            out[i, 5] = Afin[1, 1]
    else:
        for i, kp in enumerate(kps):
            out[i, :2] = kps[i]
            s = 12. * sc[i]
            a = ori[i]
            cos = math.cos(a * math.pi / 180.0)
            sin = math.sin(a * math.pi / 180.0)
            out[i, 2] = s * cos
            out[i, 3] = s * sin
            out[i, 4] = -s * sin
            out[i, 5] = s * cos
    return out


def _cmp_estimate_E_without_intrinsics(cfg,
                                       matches,
                                       kps1,
                                       kps2,
                                       calib1,
                                       calib2,
                                       img1_fname=None,
                                       img2_fname=None,
                                       scales1=None,
                                       scales2=None,
                                       ori1=None,
                                       ori2=None,
                                       A1=None,
                                       A2=None):
    '''Estimate the Essential matrix from correspondences. Computes the
    Fundamental Matrix first and then retrieves the Essential matrix assuming
    known intrinsics.
    '''

    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']

    min_matches = 8
    is_valid, matches, kp1, kp2 = _preprocess(matches, kps1, kps2, min_matches)
    if not is_valid:
        return _fail()

    if geom['method'] == 'cmp-degensac-f-laf':
        sc1 = scales1[matches[0]]
        sc2 = scales2[matches[1]]
        ang1 = ori1[matches[0]]
        ang2 = ori2[matches[1]]
        if A1 is not None:
            A1 = A1[matches[0]]
            A2 = A2[matches[1]]
        else:
            A1 = None
            A2 = None
        laf1 = get_LAF(kp1, sc1, ang1, A1)
        laf2 = get_LAF(kp2, sc2, ang2, A2)
        # print (laf1[:2])
        # print (laf2[:2])
        F, mask_F = pyransac.findFundamentalMatrix(
            laf1,
            laf2,
            geom['threshold'],
            geom['confidence'],
            geom['max_iter'],
            2.0,
            error_type=geom['error_type'],
            symmetric_error_check=True,
            enable_degeneracy_check=geom['degeneracy_check'])
    elif geom['method'] == 'cmp-degensac-f':
        F, mask_F = pyransac.findFundamentalMatrix(
            kp1,
            kp2,
            geom['threshold'],
            geom['confidence'],
            geom['max_iter'],
            0,
            error_type=geom['error_type'],
            symmetric_error_check=True,
            enable_degeneracy_check=geom['degeneracy_check'])
    elif geom['method'] == 'cmp-gc-ransac-f':
        F, mask_F = pygcransac.findFundamentalMatrix(kp1, kp2,
                                                     geom['threshold'],
                                                     geom['confidence'],
                                                     geom['max_iter'])
    elif geom['method'] == 'cmp-magsac-f':
        F, mask_F = pymagsac.findFundamentalMatrix(kp1, kp2, geom['threshold'],
                                                   geom['confidence'],
                                                   geom['max_iter'])
    else:
        raise ValueError('Unknown method: {}'.format(geom['method']))

    mask_F = mask_F.astype(bool).flatten()

    # OpenCV can return multiple values as 6x3 or 9x3 matrices
    if F is None:
        return _fail()
    elif F.shape[0] != 3:
        Fs = np.split(F, len(F) / 3)
    else:
        Fs = [F]
    if mask_F.sum() < 8:
        return _fail()

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


def _cmp_estimate_E_with_intrinsics(cfg,
                                    matches,
                                    kps1,
                                    kps2,
                                    calib1,
                                    calib2,
                                    img1_fname=None,
                                    img2_fname=None):
    '''Estimate the Essential matrix from correspondences. Assumes known
    intrinsics.
    '''

    # Reference: https://docs.opencv.org/3.4.7/d9/d0c/group__calib3d.html
    # Defalt values for confidence: 0.99 (F), 0.999 (E)
    # Default value for the reprojection threshold: 3
    # (We set them to -1 when not applicable as OpenCV complains otherwise)

    is_valid, matches, kp1, kp2 = _preprocess(matches, kps1, kps2, 5)
    if not is_valid:
        return _fail()

    # Normalize keypoints with ground truth intrinsics
    kp1_n = normalize_keypoints(kp1, calib1['K'])
    kp2_n = normalize_keypoints(kp2, calib2['K'])
    if img1_fname is not None:
        s = (cv2.imread(img1_fname)).size
        h1, w1 = s[0], s[1]
        s = (cv2.imread(img2_fname)).size
        h2, w2 = s[0], s[1]
    else:
        raise ValueError('Requires image filenames')

    cv2.setRNGSeed(cfg.opencv_seed)
    E, mask_E = pygcransac.findEssentialMatrix(kp1, kp2, calib1['K'],
                                               calib2['K'], h1, w1, h2, w2,
                                               cfg.method_geom['threshold'],
                                               cfg.method_geom['confidence'],
                                               cfg.method_geom['max_iter'])
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


def estimate_essential(cfg,
                       matches,
                       kps1,
                       kps2,
                       calib1,
                       calib2,
                       img1_fname=None,
                       img2_fname=None,
                       scales1=None,
                       scales2=None,
                       ori1=None,
                       ori2=None,
                       A1=None,
                       A2=None,
                       descs1=None,
                       descs2=None):
    '''Estimate the Essential matrix from correspondences.

    Common entry point for all methods. Currently uses cmp-pyransac to
    estimate E, with or without assuming known intrinsics.
    '''

    # Estimate E with OpenCV (assumes known intrinsics)
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']
    if geom['method'] in [
            'cmp-degensac-f', 'cmp-degensac-f-laf', 'cmp-gc-ransac-f',
            'cmp-magsac-f'
    ]:
        return _cmp_estimate_E_without_intrinsics(cfg, matches, kps1, kps2,
                                                  calib1, calib2, img1_fname,
                                                  img2_fname, scales1, scales2,
                                                  ori1, ori2, A1, A2)
    elif geom['method'] in ['cmp-gc-ransac-e']:
        return _cmp_estimate_E_with_intrinsics(cfg, matches, kps1, kps2,
                                               calib1, calib2, img1_fname,
                                               img2_fname)
    else:
        raise ValueError('Unknown method to estimate E/F')
