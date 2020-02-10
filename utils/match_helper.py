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

import itertools
import os
import cv2
import numpy as np
from utils.load_helper import load_vis
from utils.path_helper import get_match_file


def is_match_complete(cfg):
    '''Checks if match computation is complete.'''

    is_complete = os.path.exists(get_match_file(cfg))

    return is_complete


def get_matching_dist_type(cfg):
    method_match = cfg.method_dict['config_{}_{}'.format(
        cfg.dataset, cfg.task)]['matcher']
    if 'distance' in method_match:
        dist_name = method_match['distance']
        if dist_name.lower() == 'l2':
            dist = cv2.NORM_L2
        elif dist_name.lower() == 'l1':
            dist = cv2.NORM_L1
        elif dist_name.lower() == 'hamming':
            dist = cv2.NORM_HAMMING
        else:
            raise ValueError('Unknown distance', dist_name)
        return dist
    else:
        raise ValueError('Distance type is not set')


def compute_image_pairs(vis_list, num_images, vis_th, subset_index=None):
    if subset_index is None:
        vis = load_vis(vis_list)
    else:
        vis = load_vis(vis_list, subset_index)

    image_pairs = []
    for ii, jj in itertools.product(range(num_images), range(num_images)):
        if ii != jj:
            if vis[ii][jj] > vis_th:
                image_pairs.append((ii, jj))
    return image_pairs


def remove_duplicate_matches(matches, kp1, kp2):
    ''' Conveniency function to remove duplicate matches in multiple geometry
    models. This is due to methods such as SIFT that have multiple scale or
    orientation values.

    Parameters
    ----------
    matches: [2 x N] list of indices to the list of keypoints
    kp1, kp1: [(M1, M2) x 2] lists of keypoints

    Output
    -------
    unique_matches: subset of matches with duplicates removed
    '''

    if matches.size > 0:
        if matches.ndim == 1:
            matches = np.expand_dims(matches, axis=1)
        _, unique_indices = np.unique([
            np.concatenate((p1, p2))
            for p1, p2 in zip(kp1[matches[0], :2], kp2[matches[1], :2])
        ],
                                      axis=0,
                                      return_index=True)
        unique_matches = matches[:, unique_indices]
        return unique_matches
    else:
        return matches
