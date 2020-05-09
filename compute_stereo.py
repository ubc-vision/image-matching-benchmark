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
from utils.io_helper import load_h5, save_h5
from utils.load_helper import load_calib
from utils.match_helper import compute_image_pairs
from utils.path_helper import (
    get_data_path, get_fullpath_list, get_item_name_list, get_kp_file,
    get_match_file, get_geom_file, get_geom_inl_file, get_filter_match_file,
    get_stereo_depth_projection_pre_match_file,
    get_stereo_depth_projection_refined_match_file,
    get_stereo_depth_projection_final_match_file,
    get_stereo_epipolar_pre_match_file, get_stereo_epipolar_refined_match_file,
    get_stereo_epipolar_final_match_file, get_stereo_path,
    get_stereo_pose_file, get_pairs_per_threshold,
    get_repeatability_score_file)
from utils.stereo_helper import (compute_stereo_metrics_from_E,
                                 is_stereo_complete)


def main(cfg):
    '''Main function to compute matches.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    '''

    # Get data directory
    data_dir = get_data_path(cfg)

    # Load pre-computed pairs with the new visibility criteria
    pairs_per_th = get_pairs_per_threshold(data_dir)

    # Check if all files exist
    if is_stereo_complete(cfg):
        print(' -- already exists, skipping stereo eval')
        return

    # Load keypoints and matches
    keypoints_dict = load_h5(get_kp_file(cfg))
    matches_dict = load_h5(get_match_file(cfg))
    geom_dict = load_h5(get_geom_file(cfg))
    geom_inl_dict = load_h5(get_geom_inl_file(cfg))

    filter_matches_dict = load_h5(get_filter_match_file(cfg))

    # Load visiblity and images
    images_list = get_fullpath_list(data_dir, 'images')
    vis_list = get_fullpath_list(data_dir, 'visibility')
    depth_maps_list = get_fullpath_list(data_dir, 'depth_maps')
    image_names = get_item_name_list(images_list)

    # Load camera information
    calib_list = get_fullpath_list(data_dir, 'calibration')
    calib_dict = load_calib(calib_list)

    # Generate all possible pairs
    print('Generating list of all possible pairs')
    pairs = compute_image_pairs(vis_list, len(image_names), cfg.vis_th)
    print('Old pairs with the point-based visibility threshold: {} '
          '(for compatibility)'.format(len(pairs)))
    for k, v in pairs_per_th.items():
        print('New pairs at visibility threshold {}: {}'.format(k, len(v)))

    # Evaluate each stereo pair in parallel
    # Compute it for all pairs (i.e. visibility threshold 0)
    print('Compute stereo metrics for all pairs')
    #num_cores = int(multiprocessing.cpu_count() * 0.9)
    num_cores = int(len(os.sched_getaffinity(0)) * 0.9)

    result = Parallel(n_jobs=num_cores)(delayed(compute_stereo_metrics_from_E)(
        images_list[image_names.index(pair.split('-')[0])], images_list[
            image_names.index(pair.split('-')[1])], depth_maps_list[
                image_names.index(pair.split('-')[0])], depth_maps_list[
                    image_names.index(pair.split('-')[1])],
        np.asarray(keypoints_dict[pair.split('-')[0]]),
        np.asarray(keypoints_dict[pair.split('-')[1]]), calib_dict[pair.split(
            '-')[0]], calib_dict[pair.split('-')
                                 [1]], geom_dict[pair], matches_dict[pair],
        filter_matches_dict[pair], geom_inl_dict[pair], cfg)
                                        for pair in tqdm(pairs_per_th['0.0']))
    
    # Convert previous visibility list to strings
    old_keys = []
    for pair in pairs:
        old_keys.append('{}-{}'.format(image_names[pair[0]],
                                       image_names[pair[1]]))

    # Extract scores, err_q, err_t from results
    all_keys = pairs_per_th['0.0']
    err_dict, rep_s_dict = {}, {}
    geo_s_dict_pre_match, geo_s_dict_refined_match, \
        geo_s_dict_final_match = {}, {}, {}
    true_s_dict_pre_match, true_s_dict_refined_match, \
        true_s_dict_final_match = {}, {}, {}
    for i in range(len(result)):
        if all_keys[i] in old_keys:
            if result[i][5]:
                geo_s_dict_pre_match[all_keys[i]] = result[i][0][0]
                geo_s_dict_refined_match[all_keys[i]] = result[i][0][1]
                geo_s_dict_final_match[all_keys[i]] = result[i][0][2]
                true_s_dict_pre_match[all_keys[i]] = result[i][1][0]
                true_s_dict_refined_match[all_keys[i]] = result[i][1][1]
                true_s_dict_final_match[all_keys[i]] = result[i][1][2]
                err_q = result[i][2]
                err_t = result[i][3]
                rep_s_dict[all_keys[i]] = result[i][4]
                err_dict[all_keys[i]] = [err_q, err_t]
    print('Aggregating results for the old visibility constraint: '
          '{}/{}'.format(len(geo_s_dict_pre_match), len(result)))

    # Repeat with the new visibility threshold
    err_dict_th, rep_s_dict_th = {}, {}
    geo_s_dict_pre_match_th, geo_s_dict_refined_match_th, \
        geo_s_dict_final_match_th = {}, {}, {}
    true_s_dict_pre_match_th, true_s_dict_refined_match_th, \
        true_s_dict_final_match_th = {}, {}, {}
    for th, cur_pairs in pairs_per_th.items():
        _err_dict, _rep_s_dict = {}, {}
        _geo_s_dict_pre_match, _geo_s_dict_refined_match, \
            _geo_s_dict_final_match = {}, {}, {}
        _true_s_dict_pre_match, _true_s_dict_refined_match, \
            _true_s_dict_final_match = {}, {}, {}
        for i in range(len(all_keys)):
            if len(cur_pairs) > 0 and all_keys[i] in cur_pairs:
                if result[i][5]:
                    _geo_s_dict_pre_match[all_keys[i]] = result[i][0][0]
                    _geo_s_dict_refined_match[all_keys[i]] = result[i][0][1]
                    _geo_s_dict_final_match[all_keys[i]] = result[i][0][2]
                    _true_s_dict_pre_match[all_keys[i]] = result[i][1][0]
                    _true_s_dict_refined_match[all_keys[i]] = result[i][1][1]
                    _true_s_dict_final_match[all_keys[i]] = result[i][1][2]
                    err_q = result[i][2]
                    err_t = result[i][3]
                    _rep_s_dict[all_keys[i]] = result[i][4]
                    _err_dict[all_keys[i]] = [err_q, err_t]
        geo_s_dict_pre_match_th[th] = _geo_s_dict_pre_match
        geo_s_dict_refined_match_th[th] = _geo_s_dict_refined_match
        geo_s_dict_final_match_th[th] = _geo_s_dict_final_match
        true_s_dict_pre_match_th[th] = _true_s_dict_pre_match
        true_s_dict_refined_match_th[th] = _true_s_dict_refined_match
        true_s_dict_final_match_th[th] = _true_s_dict_final_match
        err_dict_th[th] = _err_dict
        rep_s_dict_th[th] = _rep_s_dict
        print('Aggregating results for threshold "{}": {}/{}'.format(
            th, len(geo_s_dict_pre_match_th[th]), len(result)))

    # Create results folder if it does not exist
    if not os.path.exists(get_stereo_path(cfg)):
        os.makedirs(get_stereo_path(cfg))

    # Finally, save packed scores and errors

    save_h5(geo_s_dict_pre_match, get_stereo_epipolar_pre_match_file(cfg))
    save_h5(geo_s_dict_refined_match,
            get_stereo_epipolar_refined_match_file(cfg))
    save_h5(geo_s_dict_final_match, get_stereo_epipolar_final_match_file(cfg))

    save_h5(true_s_dict_pre_match,
            get_stereo_depth_projection_pre_match_file(cfg))
    save_h5(true_s_dict_refined_match,
            get_stereo_depth_projection_refined_match_file(cfg))
    save_h5(true_s_dict_final_match,
            get_stereo_depth_projection_final_match_file(cfg))

    save_h5(err_dict, get_stereo_pose_file(cfg))
    save_h5(rep_s_dict, get_repeatability_score_file(cfg))
    for th in pairs_per_th:
        save_h5(geo_s_dict_pre_match_th[th],
                get_stereo_epipolar_pre_match_file(cfg, th))
        save_h5(geo_s_dict_refined_match_th[th],
                get_stereo_epipolar_refined_match_file(cfg, th))
        save_h5(geo_s_dict_final_match_th[th],
                get_stereo_epipolar_final_match_file(cfg, th))

        save_h5(true_s_dict_pre_match_th[th],
                get_stereo_depth_projection_pre_match_file(cfg, th))
        save_h5(true_s_dict_refined_match_th[th],
                get_stereo_depth_projection_refined_match_file(cfg, th))
        save_h5(true_s_dict_final_match_th[th],
                get_stereo_depth_projection_final_match_file(cfg, th))
        save_h5(err_dict_th[th], get_stereo_pose_file(cfg, th))
        save_h5(rep_s_dict_th[th], get_repeatability_score_file(cfg, th))


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(cfg)
