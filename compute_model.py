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
from time import time
import random

from collections import defaultdict
from config import get_config, print_usage
from methods import geom_models
from utils.load_helper import load_calib
from utils.io_helper import load_h5, save_h5
from utils.path_helper import (
    get_data_path, get_desc_file, get_kp_file, get_fullpath_list,
    get_scale_file, get_angle_file, get_affine_file, get_geom_inl_file,
    get_item_name_list, get_geom_file, get_match_file,
    get_filter_match_file_for_computing_model, get_geom_path,
    get_geom_cost_file, get_pairs_per_threshold)
import cv2


def compute_model(cfg,
                  matches,
                  kps1,
                  kps2,
                  calib1,
                  calib2,
                  img1_fname,
                  img2_fname,
                  scales1=None,
                  scales2=None,
                  ori1=None,
                  ori2=None,
                  A1=None,
                  A2=None,
                  descs1=None,
                  descs2=None):
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
    # Get matches through the appropriate matching module
    # For now, we consider only OpenCV
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']

    t_start = time()
    if geom['method'].startswith('cv2-'):
        model, inliers = geom_models.geom_cv2.estimate_essential(
            cfg, matches, kps1, kps2, calib1, calib2)
    elif geom['method'].startswith('skimage-'):
        model, inliers = geom_models.geom_skimage.estimate_essential(
            cfg, matches, kps1, kps2, calib1, calib2)
    elif geom['method'].startswith('cmp-'):
        model, inliers = geom_models.geom_cmp.estimate_essential(
            cfg, matches, kps1, kps2, calib1, calib2, img1_fname, img2_fname,
            scales1, scales2, ori1, ori2, A1, A2)
    elif geom['method'].startswith('intel-'):
        model, inliers = geom_models.geom_intel.estimate_essential(
            cfg, matches, kps1, kps2, calib1, calib2, scales1, scales2, ori1,
            ori2, descs1, descs2)
    else:
        raise ValueError('Unknown method to estimate E/F')

    return model, inliers, time() - t_start


def main(cfg):
    '''Main function to compute model.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    '''

    if os.path.exists(get_geom_file(cfg)):
        print(' -- already exists, skipping model computation')
        return

    # Get data directory
    keypoints_dict = load_h5(get_kp_file(cfg))

    # Load keypoints and matches
    matches_dict = load_h5(get_filter_match_file_for_computing_model(cfg))

    # Feature Matching
    print('Computing model')
    num_cores = cfg.num_opencv_threads if cfg.num_opencv_threads > 0 else int(
        len(os.sched_getaffinity(0)) * 0.9)
    # Load camera information
    data_dir = get_data_path(cfg)
    images_list = get_fullpath_list(data_dir, 'images')
    image_names = get_item_name_list(images_list)

    calib_list = get_fullpath_list(data_dir, 'calibration')
    calib_dict = load_calib(calib_list)
    pairs_per_th = get_pairs_per_threshold(data_dir)

    # Get data directory
    try:
        desc_dict = defaultdict(list)
        desc_dict = load_h5(get_desc_file(cfg))
        for k, v in desc_dict.items():
            desc_dict[k] = v
    except Exception:
        desc_dict = defaultdict(list)

    
    try:
        aff_dict = defaultdict(list)
        aff_dict1 = load_h5(get_affine_file(cfg))
        for k, v in aff_dict1.items():
            aff_dict[k] = v
    except Exception:
        aff_dict = defaultdict(list)

    try:
        ori_dict = defaultdict(list)
        ori_dict1 = load_h5(get_angle_file(cfg))
        for k, v in ori_dict1.items():
            ori_dict[k] = v
    except Exception:
        ori_dict = defaultdict(list)
    try:
        scale_dict = defaultdict(list)
        scale_dict1 = load_h5(get_scale_file(cfg))
        for k, v in scale_dict1.items():
            scale_dict[k] = v
    except Exception:
        scale_dict = defaultdict(list)

    random.shuffle(pairs_per_th['0.0'])
    result = Parallel(n_jobs=num_cores)(delayed(compute_model)(
        cfg, np.asarray(matches_dict[pair]),
        np.asarray(keypoints_dict[pair.split('-')[0]]),
        np.asarray(keypoints_dict[pair.split('-')[1]]), calib_dict[pair.split(
            '-')[0]], calib_dict[pair.split('-')[1]], images_list[
                image_names.index(pair.split('-')[0])], images_list[
                    image_names.index(pair.split('-')[1])],
        np.asarray(scale_dict[pair.split('-')[0]]),
        np.asarray(scale_dict[pair.split('-')[1]]),
        np.asarray(ori_dict[pair.split('-')[0]]),
        np.asarray(ori_dict[pair.split('-')[1]]),
        np.asarray(aff_dict[pair.split('-')[0]]),
        np.asarray(aff_dict[pair.split('-')[1]]),
        np.asarray(desc_dict[pair.split('-')[0]]),
        np.asarray(desc_dict[pair.split('-')[1]]))
                                        for pair in tqdm(pairs_per_th['0.0']))

    # Make model dictionary
    model_dict = {}
    inl_dict = {}
    timings_list = []
    for i, pair in enumerate(pairs_per_th['0.0']):
        model_dict[pair] = result[i][0]
        inl_dict[pair] = result[i][1]
        timings_list.append(result[i][2])

    # Check model directory
    if not os.path.exists(get_geom_path(cfg)):
        os.makedirs(get_geom_path(cfg))

    # Finally save packed models
    save_h5(model_dict, get_geom_file(cfg))
    save_h5(inl_dict, get_geom_inl_file(cfg))

    # Save computational cost
    save_h5({'cost': np.mean(timings_list)}, get_geom_cost_file(cfg))
    print('Geometry cost (averaged over image pairs): {:0.2f} sec'.format(
        np.mean(timings_list)))


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print(unparsed)
        print_usage()
        exit(1)

    main(cfg)
