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

import multiprocessing
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from config import get_config, print_usage
from methods import feature_matching as matching
from utils.io_helper import load_h5, save_h5
from utils.path_helper import (get_data_path, get_desc_file, get_kp_file,
                               get_match_file, get_match_path,
                               get_match_cost_file, get_pairs_per_threshold)
import cv2
WITH_FAISS=False
try:
    import faiss
    WITH_FAISS = True
except:
    pass

def compute_matches(descs1, descs2, cfg, kps1=None, kps2=None):
    '''Computes matches given descriptors.

    Parameters
    ----------
    descs1, descs2: np.ndarray
        Descriptors for the first and the second image.

    cfg: Namespace
        Configurations.

    Returns
    -------
    matches

    '''

    if cfg.num_opencv_threads > 0:
        cv2.setNumThreads(cfg.num_opencv_threads)

    # Get matches through the matching module defined in the function argument
    method_match = cfg.method_dict['config_{}_{}'.format(
        cfg.dataset, cfg.task)]['matcher']['method']
    matches, ellapsed = getattr(matching,
                                method_match).match(descs1, descs2, cfg, kps1,
                                                    kps2)

    return matches, ellapsed


def main(cfg):
    '''Main function to compute matches.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    '''

    if os.path.exists(get_match_file(cfg)):
        print(' -- already exists, skipping match computation')
        return

    # Get data directory
    data_dir = get_data_path(cfg)

    # Load pre-computed pairs with the new visibility criteria
    print('Reading list of all possible pairs')
    pairs = get_pairs_per_threshold(data_dir)['0.0']
    print('{} pre-computed pairs'.format(len(pairs)))

    # Load descriptors
    descriptors_dict = load_h5(get_desc_file(cfg))
    keypoints_dict = load_h5(get_kp_file(cfg))

    # Feature Matching
    print('Computing matches')
    num_cores = cfg.num_opencv_threads if cfg.num_opencv_threads > 0 else int(
        len(os.sched_getaffinity(0)) * 0.9)
    if WITH_FAISS:
        num_cores = min(4, num_cores)
    result = Parallel(n_jobs=num_cores)(
        delayed(compute_matches)(np.asarray(descriptors_dict[pair.split(
            '-')[0]]), np.asarray(descriptors_dict[pair.split(
                '-')[1]]), cfg, np.asarray(keypoints_dict[pair.split(
                    '-')[0]]), np.asarray(keypoints_dict[pair.split('-')[1]]))
        for pair in tqdm(pairs))

    # Make match dictionary
    matches_dict = {}
    timings_list = []
    for i, pair in enumerate(pairs):
        matches_dict[pair] = result[i][0]
        timings_list.append(result[i][1])

    # Check match directory
    if not os.path.exists(get_match_path(cfg)):
        os.makedirs(get_match_path(cfg))

    # Finally save packed matches
    save_h5(matches_dict, get_match_file(cfg))

    # Save computational cost
    save_h5({'cost': np.mean(timings_list)}, get_match_cost_file(cfg))
    print('Matching cost (averaged over image pairs): {:0.2f} sec'.format(
        np.mean(timings_list)))


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print(unparsed)
        print_usage()
        exit(1)

    main(cfg)
