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
from copy import deepcopy
from time import time
from utils.match_helper import get_matching_dist_type, remove_duplicate_matches
WITH_FAISS = False
try:
    import faiss
    WITH_FAISS = True
except:
    pass

def match(desc1, desc2, cfg, kps1=None, kps2=None):
    ''' Computes and returns matches with the ratio test

    param desc1: descriptors of the first image
    param desc2: descriptors of the second image
    param cfg: Configuration

    return matches: np.ndarray with match indices
    '''

    # Parse options
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    method_match = cfg.method_dict[cur_key]['matcher']
    filter_type = method_match['filtering']['type'].lower()

    # Ratio test threshold
    if filter_type.lower() in [
            'snn_ratio_pairwise', 'snn_ratio_vs_last', 'fginn_ratio_pairwise'
    ]:
        ratio_th = method_match['filtering']['threshold']
        if ratio_th < 0.1 or ratio_th > 1.01:
            raise ValueError('Ratio test threshold outside expected range')

    # FGINN spatial threshold
    if filter_type in ['fginn_ratio_pairwise']:
        fginn_spatial_th = method_match['filtering']['fginn_radius']
        if fginn_spatial_th < 0 or fginn_spatial_th > 500:
            raise ValueError('FGINN radius outside expected range')

    # Distance threshold
    max_dist = None
    if 'descriptor_distance_filter' in method_match:
        max_dist = method_match['descriptor_distance_filter']['threshold']

    # Skip this if there are no features
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.empty((2, 0)), 0

    # Reformat descriptors according to the distance type
    dist = get_matching_dist_type(cfg)
    if method_match['distance'].lower() == 'hamming':
        desc1 = desc1.astype(np.uint8)
        desc2 = desc2.astype(np.uint8)
    elif method_match['distance'].lower() in ['l1', 'l2']:
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
    else:
        raise ValueError('Unknown distance type')

    t_start = time()

    # Use opencv BF/FLANN matcher
    do_flann = method_match['flann']

    if do_flann:
        if dist == cv2.NORM_L2:
            FLANN_INDEX_KDTREE = 1  # FLANN_INDEX_KDTREE
        elif dist == cv2.NORM_HAMMING:
            FLANN_INDEX_KDTREE = 5  # FLANN_INDEX_HIERARCHICAL
        else:
            FLANN_INDEX_KDTREE = 0  # FLANN_INDEX_LINEAR
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=128)  # or pass empty dictionary
        bf = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        bf = cv2.BFMatcher(dist, crossCheck=False)

    # Matching
    num_nn = method_match['num_nn']
    try:
        if WITH_FAISS and dist == cv2.NORM_L2:
            #print ("FAISS")
            dbsize, dim = desc2.shape
            res = faiss.StandardGpuResources()  # use a single GPU
            index_flat = faiss.IndexFlatL2(dim)  # build a flat (CPU) index
            nn = faiss.index_cpu_to_gpu(res, 0, index_flat)
            nn.add(desc2)         # add vectors to the index
            if 'fginn' in filter_type:
                k = 10 + num_nn
            else:
                k=max(2, num_nn + 1)
            dists,idx = nn.search(desc1, k)
            matches = []
            #print (dists.shape)
            for query_idx, (dd, ii) in enumerate(zip(dists, idx)):
                cur_match = []
                for db_idx, m_dist in zip(ii,dd):
                    cur_match.append(cv2.DMatch(query_idx, db_idx, m_dist))
                matches.append(cur_match)
        else:
            if 'fginn' in filter_type:
                matches = bf.knnMatch(desc1, desc2, k=10 + num_nn)
            else:
                matches = bf.knnMatch(desc1, desc2, k=max(2, num_nn + 1))
    except:
        print ("FAISS crashed, fallback to opnecv")
        if 'fginn' in filter_type:
            matches = bf.knnMatch(desc1, desc2, k=10 + num_nn)
        else:
            matches = bf.knnMatch(desc1, desc2, k=max(2, num_nn + 1))
    # Apply filtering (ratio test or something else)
    valid_matches = []

    if filter_type == 'none':
        for cur_match in matches:
            tmp_valid_matches = [nn_1 for nn_1 in cur_match[:-1]]
            valid_matches.extend(tmp_valid_matches)
    elif filter_type == 'snn_ratio_pairwise':
        for cur_match in matches:
            tmp_valid_matches = [
                nn_1 for nn_1, nn_2 in zip(cur_match[:-1], cur_match[1:])
                if nn_1.distance <= ratio_th * nn_2.distance
            ]
            valid_matches.extend(tmp_valid_matches)
    elif filter_type == 'snn_ratio_vs_last':
        for cur_match in matches:
            nn_n = cur_match[-1]
            tmp_valid_matches = [
                nn_i for nn_i in cur_match[:-1]
                if nn_i.distance <= ratio_th * nn_n.distance
            ]
            valid_matches.extend(tmp_valid_matches)
    elif filter_type == 'fginn_ratio_pairwise':
        flann = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        not_fginns = flann.radiusMatch(kps2.astype(np.float32),
                                       kps2.astype(np.float32),
                                       fginn_spatial_th)
        for m_idx, cur_match in enumerate(matches):
            for mii in range(num_nn):
                cur_non_fginns = [
                    x.trainIdx for x in not_fginns[cur_match[mii].trainIdx]
                ]
                for nn_2 in cur_match[mii + 1:]:
                    if cur_match[mii].distance <= ratio_th * nn_2.distance:
                        valid_matches.append(cur_match[mii])
                        break
                    if nn_2.trainIdx not in cur_non_fginns:
                        break
    else:
        raise ValueError('Unknown filter type')

    # Filter matches by descriptor distance
    if max_dist:
        valid_matches_with_dist = []
        for valid_match in valid_matches:
            if valid_match.distance <= max_dist:
                valid_matches_with_dist.append(valid_match)
        valid_matches = valid_matches_with_dist

    # Turn opencv return format into numpy
    matches_list = []
    for m in valid_matches:
        matches_list.append([m.queryIdx, m.trainIdx])
    matches = np.asarray(matches_list).T

    # If two-way matching is enabled
    if method_match['symmetric']['enabled']:
        reduce_method = method_match['symmetric']['reduce'].lower()

        # Hacky but works: disable symmetric matching and call self
        cfg_temp = deepcopy(cfg)
        cfg_temp.method_dict['config_{}_{}'.format(
            cfg.dataset, cfg.task)]['matcher']['symmetric']['enabled'] = False
        matches_other_direction, ellapsed_other_direction = match(
            desc2, desc1, cfg_temp, kps2, kps1)

        if reduce_method == 'any':
            if len(matches.shape) == 2 and \
               len(matches_other_direction.shape) == 2:
                matches = np.concatenate(
                    [matches, matches_other_direction[::-1]], axis=1)
        elif reduce_method == 'both':
            m1 = matches.T
            m2 = matches_other_direction.T
            out = []
            if len(m2) < 2:
                matches = np.zeros((2, 0)).astype(np.int32)
            else:
                for i in range(len(m1)):
                    i1, i2 = m1[i]
                    mask = m2[:, 0] == i2
                    row = m2[mask]
                    if len(row) > 0:
                        i22, i11 = row[0]
                        if (i1 == i11) and (i2 == i22):
                            out.append(m1[i].reshape(1, -1))
                if len(out) > 0:
                    matches = np.concatenate(out, axis=0).T.astype(np.int32)
                else:
                    matches = np.zeros((2, 0)).astype(np.int32)
        else:
            raise ValueError('Unknown symmetrical match reduce ',
                             reduce_method)
    else:
        ellapsed_other_direction = 0

    # Remove duplicate matches
    matches = remove_duplicate_matches(matches, kps1, kps2)

    return matches, time() - t_start + ellapsed_other_direction
