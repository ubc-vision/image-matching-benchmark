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

import numpy as np
import math
import cv2

# Now also available in Pypi:
# https://pypi.org/project/extract-patches/0.1.0/


def extract_patches(kpts, img, PS=32, mag_factor=10.0, input_format='cv2'):
    '''
    Extracts patches given the keypoints in the one of the following formats:
     - cv2: list of cv2 keypoints
     - cv2+A: tuple of (list of cv2 keypoints, Nx2x2 np array)
     - ellipse: Nx5 np array, single row is [x y a b c]
     - xyA: Nx6 np array, single row is [x y a11 a12 a21 a22]
     - LAF: Nx2x3 np array, single row is [a11 a12 x; a21 a22 y]

    Returns list of patches.

    Upgraded version of mag_factor is a scale coefficient. Use 10 for
    extracting OpenCV SIFT patches, 1.0 for OpenCV ORB patches, etc.
    PS is the output patch size in pixels
    '''

    if input_format == 'cv2':
        Ms, pyr_idxs = convert_cv2_keypoints(kpts, PS, mag_factor)
    elif input_format == 'cv2+A':
        Ms, pyr_idxs = convert_cv2_plus_A_keypoints(kpts[0], kpts[1], PS,
                                                    mag_factor)
    elif (input_format == 'ellipse') or (input_format == 'xyabc'):
        assert kpts.shape[1] == 5
        Ms, pyr_idxs = convert_ellipse_keypoints(kpts, PS, mag_factor)
    elif input_format == 'xyA':
        assert kpts.shape[1] == 6
        Ms, pyr_idxs = convert_xyA(kpts, PS, mag_factor)
    elif input_format == 'LAF':
        assert len(kpts.shape) == 3
        assert len(kpts.shape[2]) == 3
        assert len(kpts.shape[1]) == 2
        Ms, pyr_idxs = convert_LAFs(kpts, PS, mag_factor)
    else:
        raise ValueError('Unknown input format', input_format)

    return extract_patches_Ms(Ms, img, pyr_idxs, PS)


def build_image_pyramid(img, min_size):
    '''Builds image pyramid.'''

    patches = []
    img_pyr = [img]
    cur_img = img
    while np.min(cur_img.shape[:2]) > min_size:
        cur_img = cv2.pyrDown(cur_img)
        img_pyr.append(cur_img)
    return img_pyr


def extract_patches_Ms(Ms, img, pyr_idxs=[], PS=32):
    '''
    Builds image pyramid and rectifies patches around keypoints
    in the tranformation matrix format
    from the appropriate level of image pyramid,
    removing high freq artifacts. Border mode is set to 'replicate',
    so the boundary patches don`t have crazy black borders

    Returns list of patches.
    '''

    assert len(Ms) == len(pyr_idxs)
    img_pyr = build_image_pyramid(img, PS / 2.0)
    patches = []
    for i, M in enumerate(Ms):
        patch = cv2.warpAffine(img_pyr[pyr_idxs[i]], M, (PS, PS),
                             flags=cv2.WARP_INVERSE_MAP + \
                             cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
                             borderMode=cv2.BORDER_REPLICATE)
        patches.append(patch)
    return patches


def convert_cv2_keypoints(kps, PS, mag_factor):
    '''
    Converts OpenCV keypoints into transformation matrix
    and pyramid index to extract from for the patch extraction
    '''

    Ms = []
    pyr_idxs = []
    for i, kp in enumerate(kps):
        x, y = kp.pt
        s = kp.size
        a = kp.angle
        s = mag_factor * s / PS
        pyr_idx = int(math.log(s, 2))
        d_factor = float(math.pow(2., pyr_idx))
        s_pyr = s / d_factor
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)
        M = np.matrix([[
            +s_pyr * cos, -s_pyr * sin,
            (-s_pyr * cos + s_pyr * sin) * PS / 2.0 + x / d_factor
        ],
                       [
                           +s_pyr * sin, +s_pyr * cos,
                           (-s_pyr * sin - s_pyr * cos) * PS / 2.0 +
                           y / d_factor
                       ]])
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs


def convert_cv2_plus_A_keypoints(kps, A, PS, mag_factor):
    '''
    Converts OpenCV keypoints + A [n x 2 x 2] affine shape
    into transformation matrix
    and pyramid index to extract from for the patch extraction
    '''

    Ms = []
    pyr_idxs = []
    for i, kp in enumerate(kps):
        x, y = kp.pt
        s = kp.size
        a = kp.angle
        s = mag_factor * s / PS
        pyr_idx = int(math.log(s, 2))
        d_factor = float(math.pow(2., pyr_idx))
        s_pyr = s / d_factor
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)
        Ai = A[i]
        RotA = np.matrix([[+s_pyr * cos, -s_pyr * sin],
                          [+s_pyr * sin, +s_pyr * cos]])
        Ai = np.matmul(RotA, np.matrix(Ai))
        M = np.concatenate([
            Ai,
            [[(-Ai[0, 0] - Ai[0, 1]) * PS / 2.0 + x / d_factor],
             [(-Ai[1, 0] - Ai[1, 1]) * PS / 2.0 + y / d_factor]]
        ],
                           axis=1)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs


def convert_xyA(kps, PS, mag_factor):
    '''
    Converts N x [x y a11 a12 a21 a22] affine regions
    into a transformation matrix
    and pyramid index to extract from for the patch extraction
    '''

    Ms = []
    pyr_idxs = []
    for i, kp in enumerate(kps):
        x = kp[0]
        y = kp[1]
        Ai = mag_factor * kp[2:].reshape(2, 2) / PS
        s = np.sqrt(np.abs(Ai[0, 0] * Ai[1, 1] - Ai[0, 1] * Ai[1, 0]))
        pyr_idx = int(math.log(s, 2))
        d_factor = float(math.pow(2., pyr_idx))
        Ai = Ai / d_factor
        M = np.concatenate([
            Ai,
            [[(-Ai[0, 0] - Ai[0, 1]) * PS / 2.0 + x / d_factor],
             [(-Ai[1, 0] - Ai[1, 1]) * PS / 2.0 + y / d_factor]]
        ],
                           axis=1)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)

    return Ms, pyr_idxs


def convert_LAFs(kps, PS, mag_factor):
    '''
    Converts N x [a11 a12 x; a21 a22 y] affine regions
    into a transformation matrix
    and pyramid index to extract from for the patch extraction
    '''

    Ms = []
    pyr_idxs = []
    for i, kp in enumerate(kps):
        x = kp[0, 2]
        y = kp[1, 2]
        Ai = mag_factor * kp[:2, :2] / PS
        s = np.sqrt(np.abs(Ai[0, 0] * Ai[1, 1] - Ai[0, 1] * Ai[1, 0]))
        pyr_idx = int(math.log(s, 2))
        d_factor = float(math.pow(2., pyr_idx))
        Ai = Ai / d_factor
        M = np.concatenate([
            Ai,
            [[(-Ai[0, 0] - Ai[0, 1]) * PS / 2.0 + x / d_factor],
             [(-Ai[1, 0] - Ai[1, 1]) * PS / 2.0 + y / d_factor]]
        ],
                           axis=1)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs


def Ell2LAF(ell):
    '''
    Converts ellipse [x y a b c] into a [a11 a12 x; a21 a22 y] affine region
    '''

    A23 = np.zeros((2, 3))
    A23[0, 2] = ell[0]
    A23[1, 2] = ell[1]
    a = ell[2]
    b = ell[3]
    c = ell[4]
    sc = np.sqrt(np.sqrt(a * c - b * b))
    ia, ib, ic = invSqrt(
        a, b, c)  #because sqrtm returns ::-1, ::-1 matrix, don`t know why
    A = np.array([[ia, ib], [ib, ic]]) / sc
    sc = np.sqrt(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])
    A23[0:2, 0:2] = rectifyAffineTransformationUpIsUp(A / sc) * sc
    return A23


def invSqrt(a, b, c):
    eps = 1e-12
    mask = (b != 0)
    r1 = mask * (c - a) / (2. * b + eps)
    t1 = np.sign(r1) / (np.abs(r1) + np.sqrt(1. + r1 * r1))
    r = 1.0 / np.sqrt(1. + t1 * t1)
    t = t1 * r

    r = r * mask + 1.0 * (1.0 - mask)
    t = t * mask

    x = 1. / np.sqrt(r * r * a - 2 * r * t * b + t * t * c)
    z = 1. / np.sqrt(t * t * a + 2 * r * t * b + r * r * c)

    d = np.sqrt(x * z)

    x = x / d
    z = z / d

    new_a = r * r * x + t * t * z
    new_b = -r * t * x + t * r * z
    new_c = t * t * x + r * r * z

    return new_a, new_b, new_c


def rectifyAffineTransformationUpIsUp(A):
    '''
    Sets [a11 a12; a21 a22] into upright orientation
    '''

    det = np.sqrt(np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1] + 1e-10))
    b2a2 = np.sqrt(A[0, 1] * A[0, 1] + A[0, 0] * A[0, 0])
    A_new = np.zeros((2, 2))
    A_new[0, 0] = b2a2 / det
    A_new[0, 1] = 0
    A_new[1, 0] = (A[1, 1] * A[0, 1] + A[1, 0] * A[0, 0]) / (b2a2 * det)
    A_new[1, 1] = det / b2a2
    return A_new


def convert_ellipse_keypoints(ells, PS, mag_factor):
    '''
    Converts N x [x y a b c] affine regions
    into transformation matrix
    and pyramid index to extract from for the patch extraction
    '''

    Ms = []
    pyr_idxs = []
    for i, ell in enumerate(ells):
        LAF = Ell2LAF(ell)
        x = LAF[0, 2]
        y = LAF[1, 2]
        Ai = mag_factor * LAF[:2, :2] / PS
        s = np.sqrt(np.abs(Ai[0, 0] * Ai[1, 1] - Ai[0, 1] * Ai[1, 0]))
        pyr_idx = int(math.log(s, 2))
        d_factor = float(math.pow(2., pyr_idx))
        Ai = Ai / d_factor
        M = np.concatenate([
            Ai,
            [[(-Ai[0, 0] - Ai[0, 1]) * PS / 2.0 + x / d_factor],
             [(-Ai[1, 0] - Ai[1, 1]) * PS / 2.0 + y / d_factor]]
        ],
                           axis=1)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)

    return Ms, pyr_idxs
