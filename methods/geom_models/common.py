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
