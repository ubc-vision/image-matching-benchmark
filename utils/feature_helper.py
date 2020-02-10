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

import cv2
import numpy as np

from utils.path_helper import get_desc_file, get_kp_file

# ----------------------------------------------------------------------
# Global constants

# Keypoint List Structure Index Info
IDX_X, IDX_Y, IDX_SIZE, IDX_ANGLE, IDX_RESPONSE, IDX_OCTAVE = (
    0, 1, 2, 3, 4, 5)  # , IDX_CLASSID not used
IDX_a, IDX_b, IDX_c = (6, 7, 8)

# NOTE the row-major colon-major adaptation here
IDX_A0, IDX_A2, IDX_A1, IDX_A3 = (9, 10, 11, 12)

# # IDX_CLASSID for KAZE
# IDX_CLASSID = 13

KP_LIST_LEN = 13
# ----------------------------------------------------------------------


def l_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


def update_affine(kp):
    '''Returns an updated version of the keypoint.
    Note
    ----
    This function should be applied only to individual keypoints, not a list.
    '''

    # Compute A0, A1, A2, A3
    S = np.asarray([[kp[IDX_a], kp[IDX_b]], [kp[IDX_b], kp[IDX_c]]])
    invS = np.linalg.inv(S)
    a = np.sqrt(invS[0, 0])
    b = invS[0, 1] / max(a, 1e-18)
    A = np.asarray([[a, 0], [b, np.sqrt(max(invS[1, 1] - b**2, 0))]])

    # We need to rotate first!
    cos_val = np.cos(np.deg2rad(kp[IDX_ANGLE]))
    sin_val = np.sin(np.deg2rad(kp[IDX_ANGLE]))
    R = np.asarray([[cos_val, -sin_val], [sin_val, cos_val]])

    A = np.dot(A, R)

    kp[IDX_A0] = A[0, 0]
    kp[IDX_A1] = A[0, 1]
    kp[IDX_A2] = A[1, 0]
    kp[IDX_A3] = A[1, 1]

    return kp


def kp_list_2_opencv_kp_list(kp_list):
    '''Converts our kp list structure into opencv keypoints.

    Note that the size is now diameter.
    '''

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(
            x=kp[IDX_X],
            y=kp[IDX_Y],
            _size=kp[IDX_SIZE] * 2.0,
            _angle=kp[IDX_ANGLE],
            _response=kp[IDX_RESPONSE],
            _octave=np.int32(kp[IDX_OCTAVE]),
            # _class_id=np.int32(kp[IDX_CLASSID])
        )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list


def opencv_kp_list_2_kp_list(opencv_kp_list):
    '''Converts opencv keypoints into the kp list structure.

    Note that the size is now radius.
    '''

    kp_list = []
    for opencv_kp in opencv_kp_list:
        kp = np.zeros((KP_LIST_LEN, ))
        kp[IDX_X] = opencv_kp.pt[0]
        kp[IDX_Y] = opencv_kp.pt[1]
        kp[IDX_SIZE] = opencv_kp.size * 0.5
        kp[IDX_ANGLE] = opencv_kp.angle
        kp[IDX_RESPONSE] = opencv_kp.response
        kp[IDX_OCTAVE] = opencv_kp.octave

        # Compute a,b,c for vgg affine
        kp[IDX_a] = 1. / (kp[IDX_SIZE]**2)
        kp[IDX_b] = 0.
        kp[IDX_c] = 1. / (kp[IDX_SIZE]**2)

        # Compute A0, A1, A2, A3 and update
        kp = update_affine(kp)
        # kp[IDX_CLASSID] = opencv_kp.class_id

        kp_list += [kp]

    return kp_list


def convert_opencv_kp_desc(kp, desc, num_kp):
    '''Converts opencv keypoints and descriptors to benchmark format.

    Parameters
    ----------
    kp: list
        List of keypoints in opencv format
    desc: list
        List of descriptors in opencv format
    num_kp: int
        Number of keypoints to extract per image
    '''

    # Convert OpenCV keypoints to list data structure used for the benchmark.
    kp = opencv_kp_list_2_kp_list(kp)

    # Sort keypoints and descriptors by keypoints response
    kp_desc = [(_kp, _desc)
               for _kp, _desc in sorted(zip(kp, desc),
                                        key=lambda x: x[0][IDX_RESPONSE])]
    kp_sorted = [kp for kp, desc in kp_desc]
    desc_sorted = [desc for kp, desc in kp_desc]
    # Reverse for descending order
    keypoints = kp_sorted[::-1]
    descriptors = desc_sorted[::-1]
    # Remove redundant points
    cur_num_kp = len(keypoints)
    keypoints = keypoints[:min(cur_num_kp, num_kp)]
    descriptors = descriptors[:min(cur_num_kp, num_kp)]

    return keypoints, descriptors


def is_feature_complete(cfg):
    """Checks if feature extraction is complete."""

    # is_complete = os.path.exists(get_kp_file(cfg)) and os.path.exists(
    #     get_desc_file(cfg))
    is_complete = os.path.exists(get_kp_file(cfg)) 
    return is_complete
