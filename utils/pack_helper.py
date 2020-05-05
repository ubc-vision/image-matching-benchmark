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

import datetime
import os
from copy import deepcopy
from collections import OrderedDict
import numpy as np

from third_party.colmap.scripts.python.read_write_model import (
        read_model, qvec2rotmat)
from utils.colmap_helper import (get_best_colmap_index,
                                 get_colmap_image_path_list,
                                 get_colmap_output_path, get_colmap_pose_file,
                                 get_colmap_image_path_list,
                                 valid_bag)
from utils.io_helper import load_json
from utils.path_helper import (
    get_data_path, get_fullpath_list, get_item_name_list, get_kp_file,
    get_stereo_depth_projection_pre_match_file,
    get_stereo_depth_projection_refined_match_file,
    get_stereo_depth_projection_final_match_file,
    get_stereo_epipolar_pre_match_file, get_stereo_epipolar_refined_match_file,
    get_stereo_epipolar_final_match_file, get_stereo_pose_file,
    get_repeatability_score_file, get_pairs_per_threshold, get_match_cost_file,
    get_filter_cost_file, get_geom_cost_file, get_desc_file,
    get_filter_match_file)
from utils.eval_helper import (ate_ransac, calc_std, calc_max_trans_error,
                               calc_num_iter_ransac)
from utils.load_helper import load_calib, load_h5_valid_image

def get_current_date():
    curr_date = datetime.datetime.now()
    curr_date = format(curr_date.year, '04')[2:] + '-' + \
        format(curr_date.month, '02') + '-' + format(curr_date.day, '02')
    return curr_date

def get_num_in_bag(cfg):
    '''Retrieve number of bags per subset from the json file.'''

    scene_list = load_json(
        getattr(cfg, 'scenes_{}_{}'.format(cfg.dataset, cfg.subset)))
    bag_size_json = load_json(
        getattr(cfg, 'splits_{}_{}'.format(cfg.dataset, cfg.subset)))
    bag_size_list = [b['bag_size'] for b in bag_size_json]
    bag_size_num = [b['num_in_bag'] for b in bag_size_json]

    return bag_size_num[bag_size_list.index(cfg.bag_size)]


def compute_avg_num_keypoints(res_dict, deprecated_images, cfg):
    '''Compute the average number of keypoints and add it to the dictionary.'''

    # Load keypoints file
    keypoints_dict = load_h5_valid_image(
        get_kp_file(cfg),deprecated_images)

    # Simply return average of all keypoints per image
    num_kp_list = []
    for key, values in keypoints_dict.items():
        num_kp_list += [len(values)]

    # Now compute average number of keypoints
    res_dict['avg_num_keypoints'] = float(np.mean(num_kp_list))


def compute_num_inliers(res_dict, deprecated_images, cfg):
    '''Compile match numbers at different stages into the dictionary.'''

    # if cfg.method_dict['config_{}_{}'.format(cfg.dataset,
    #                                          cfg.task)]['use_custom_matches']:
    #     raise NotImplementedError(
    #         'Probably read only once? What to do with runs?')

    # Load pre-computed pairs with the new visibility criteria
    data_dir = get_data_path(cfg)
    pairs_per_th = get_pairs_per_threshold(data_dir)
    stereo_thresholds = list(pairs_per_th.keys())
    epipolar_err_dict = {}

    # Load epipolar error file
    for th in [None] + stereo_thresholds:
        epipolar_err_dict['matcher'] = load_h5_valid_image(
            get_stereo_epipolar_pre_match_file(cfg, th) ,deprecated_images)
        epipolar_err_dict['filter'] = load_h5_valid_image(
            get_stereo_epipolar_refined_match_file(cfg, th) ,deprecated_images)
        epipolar_err_dict['geom'] = load_h5_valid_image(
            get_stereo_epipolar_final_match_file(cfg, th) ,deprecated_images)

        for key_stage, values1 in epipolar_err_dict.items():
            # Simply return average of all pairs
            num_matches = []
            for key, values2 in values1.items():
                num_matches.append(len(values2))

            # Save the number of inliers
            vis_label = '' if th is None else '_th_{}'.format(th)
            res_dict['num_matches_{}{}'.format(key_stage, vis_label)] = float(
                np.mean(num_matches) if len(num_matches) > 0 else 0)


def compute_timings(res_dict, deprecated_images, cfg):
    '''Compile timings (if any) into the dictionary.'''

    # Load cost files
    try:
        cost_match = load_h5_valid_image(
            get_match_cost_file(cfg) ,deprecated_images)
        res_dict['match_cost'] = cost_match['cost']
    except Exception:
        res_dict['match_cost'] = 0

    try:
        cost_filter = load_h5_valid_image(
            get_filter_cost_file(cfg) ,deprecated_images)
        res_dict['filter_cost'] = cost_filter['cost']
    except Exception:
        res_dict['filter_cost'] = 0

    try:
        cost_geom = load_h5_valid_image(
            get_geom_cost_file(cfg) ,deprecated_images)
        res_dict['geom_cost'] = cost_geom['cost']
    except Exception:
        res_dict['geom_cost'] = 0


def compute_matching_scores_epipolar(res_dict, deprecated_images, cfg):
    '''Compute Matching Scores (with calib) and add them to the dictionary.'''

    # Load pre-computed pairs with the new visibility criteria
    data_dir = get_data_path(cfg)
    pairs_per_th = get_pairs_per_threshold(data_dir)
    stereo_thresholds = list(pairs_per_th.keys())
    epipolar_err_dict = {}

    # Load epipolar error file
    values = {}
    for th in [None] + stereo_thresholds:
        # Init empty list
        epipolar_err_dict['pre_match'] = load_h5_valid_image(
            get_stereo_epipolar_pre_match_file(cfg, th) ,deprecated_images)
        epipolar_err_dict['refined_match'] = load_h5_valid_image(
            get_stereo_epipolar_refined_match_file(cfg, th) ,deprecated_images)
        epipolar_err_dict['final_match'] = load_h5_valid_image(
            get_stereo_epipolar_final_match_file(cfg, th) ,deprecated_images)

        for key_stage, values1 in epipolar_err_dict.items():
            if key_stage not in values:
                values[key_stage] = []

            # Simply return average of all pairs
            ms_list = []
            for key, values2 in values1.items():
                if len(values2) > 0:
                    ms_list += [
                        np.mean(
                            values2 < cfg.matching_score_epipolar_threshold)
                    ]
                else:
                    ms_list += [0]

            # Now compute average number of keypoints
            vis_label = '' if th is None else '_th_{}'.format(th)
            values[key_stage].append(
                float(np.mean(ms_list) if len(ms_list) > 0 else 0))


def compute_matching_scores_depth_projection(res_dict, deprecated_images, cfg):
    '''Compute matching scores (with depth) and add them to the dictionary.'''
    px_th_list = cfg.matching_score_and_repeatability_px_threshold

    # Load pre-computed pairs with the new visibility criteria
    data_dir = get_data_path(cfg)
    pairs_per_th = get_pairs_per_threshold(data_dir)
    stereo_thresholds = list(pairs_per_th.keys())
    reprojection_err_dict = {}

    # Load epipolar error file
    for th in [None] + stereo_thresholds:
        reprojection_err_dict['pre_match'] = load_h5_valid_image(
            get_stereo_depth_projection_pre_match_file(cfg, th) ,deprecated_images)
        reprojection_err_dict['refined_match'] = load_h5_valid_image(
            get_stereo_depth_projection_refined_match_file(cfg, th) ,deprecated_images)
        reprojection_err_dict['final_match'] = load_h5_valid_image(
            get_stereo_depth_projection_final_match_file(cfg, th) ,deprecated_images)
        for key_stage, values1 in reprojection_err_dict.items():
            acc = []
            for px_th in px_th_list:
                ms = []
                # Simply return average of all pairs
                for key, values2 in values1.items():
                    if len(values2) > 0:
                        ms += [np.mean(values2 < px_th)]
                    else:
                        ms += [0]
                acc += [float(np.mean(ms) if len(ms) > 0 else 0)]

            # Now compute average number of keypoints
            vis_label = '' if th is None else '_th_{}'.format(th)
            res_dict['matching_scores_depth_projection_{}{}'.format(
                key_stage, vis_label)] = acc


def compute_repeatability(res_dict, deprecated_images, cfg):
    '''Compute repeatability and add it to the dictionary.'''
    px_th_list = cfg.matching_score_and_repeatability_px_threshold

    # Load pre-computed pairs with the new visibility criteria
    data_dir = get_data_path(cfg)
    pairs_per_th = get_pairs_per_threshold(data_dir)
    stereo_thresholds = list(pairs_per_th.keys())

    # Load epipolar error file
    for th in [None] + stereo_thresholds:
        ms_list_list = [[] for i in range(len(px_th_list))]
        repeatability_dict = load_h5_valid_image(
            get_repeatability_score_file(cfg, th) ,deprecated_images)

        for key, values in repeatability_dict.items():
            # Simply return average of all pairs
            for idx in range(len(px_th_list)):
                ms_list_list[idx] += [values[idx]]

        # Now compute average number of keypoints
        acc = []
        for px_th, ms_list in zip(px_th_list, ms_list_list):
            acc += [float(np.mean(ms_list) if len(ms_list) > 0 else 0)]
        vis_label = '' if th is None else '_th_{}'.format(th)
        res_dict['repeatability{}'.format(vis_label)] = acc


def compute_qt_auc(res_dict, deprecated_images, cfg):
    '''Compute pose accuracy (stereo) and add it to the dictionary.'''

    # Load pre-computed pairs with the new visibility criteria
    data_dir = get_data_path(cfg)
    pairs_per_th = get_pairs_per_threshold(data_dir)
    stereo_thresholds = list(pairs_per_th.keys())

    # Load pose error for stereo
    for th in [None] + stereo_thresholds:
        pose_err_dict = load_h5_valid_image(
            get_stereo_pose_file(cfg, th) ,deprecated_images)

        # Gather err_q, err_t
        err_qt = []
        for key, value in pose_err_dict.items():
            err_qt += [value]

        if len(err_qt) > 0:
            err_qt = np.asarray(err_qt)
            # Take the maximum among q and t errors
            err_qt = np.max(err_qt, axis=1)
            # Convert to degree
            err_qt = err_qt * 180.0 / np.pi
            # Make infs to a large value so that np.histogram can be used.
            err_qt[err_qt == np.inf] = 1e6

            # Create histogram
            bars = np.arange(11)
            qt_hist, _ = np.histogram(err_qt, bars)
            # Normalize histogram with all possible pairs
            num_pair = float(len(err_qt))
            qt_hist = qt_hist.astype(float) / num_pair

            # Make cumulative
            qt_acc = np.cumsum(qt_hist)
        else:
            qt_acc = [0] * 10

        # Save to dictionary
        label = '' if th is None else '_th_{}'.format(th)
        res_dict['qt_01_10{}'.format(label)] = qt_hist.tolist()
        res_dict['qt_auc_05{}'.format(label)] = np.mean(qt_acc[:5])
        res_dict['qt_auc_10{}'.format(label)] = np.mean(qt_acc)


def compute_qt_auc_colmap(res_dict, deprecated_images, cfg):
    '''Compute pose accuracy (multiview) and add it to the dictionary.'''

    qt_acc_list = []
    # For all the bags
    cfg_bag = deepcopy(cfg)
    qt_hist_list = np.empty([0, 10])
    for bag_id in range(get_num_in_bag(cfg_bag)):
        cfg_bag.bag_id = bag_id
        
        # Skip if bag contains deprecated images
        if not valid_bag(cfg_bag, deprecated_images):
            continue

        # Load pose error for colmap
        pose_err_dict = load_h5_valid_image(
            get_colmap_pose_file(cfg_bag) ,deprecated_images)

        # Gather err_q, err_t
        err_qt = []
        for key, value in pose_err_dict.items():
            err_qt += [value]
        err_qt = np.asarray(err_qt)

        # Take the maximum among q and t errors
        err_qt = np.max(err_qt, axis=1)

        # Convert to degree
        err_qt = err_qt * 180.0 / np.pi

        # Make infs to a large value so that np.histogram can be used.
        err_qt[err_qt == np.inf] = 1e6

        # Create histogram
        bars = np.arange(11)
        qt_hist, _ = np.histogram(err_qt, bars)
        # Normalize histogram with all possible pairs (note that we already
        # have error results filled in with infs if necessary, thus we don't
        # have to worry about them)
        num_pair = float(len(err_qt))
        qt_hist = qt_hist.astype(float) / num_pair

        # Make cumulative and store to list
        qt_acc_list += [np.cumsum(qt_hist)]
        qt_hist_list = np.concatenate(
            (qt_hist_list, np.expand_dims(qt_hist, axis=0)), axis=0)

    # Aggregate all results
    qt_acc = np.mean(qt_acc_list, axis=0)
    qt_hist = np.squeeze(np.mean(qt_hist_list, axis=0))

    # Save to dictionary
    res_dict['qt_colmap_01_10'] = qt_hist.tolist()
    res_dict['qt_auc_colmap_05'] = np.mean(qt_acc[:5])
    res_dict['qt_auc_colmap_10'] = np.mean(qt_acc)


def compute_ATE(res_dict, deprecated_images, cfg):
    '''Compute the Absolute Trajectory Error and add it to the dictionary.'''

    ate_list = []

    # For all the bags
    cfg_bag = deepcopy(cfg)
    data_dir = get_data_path(cfg)
    calib_list = get_fullpath_list(data_dir, 'calibration')
    calib_dict = load_calib(calib_list)
    for bag_id in range(get_num_in_bag(cfg_bag)):
        cfg_bag.bag_id = bag_id

        # Skip if bag contains deprecated images
        if not valid_bag(cfg_bag, deprecated_images):
            continue

        # Get colmap output (binary files) path
        colmap_res_path = os.path.join(get_colmap_output_path(cfg_bag),
                                       str(get_best_colmap_index(cfg_bag)))

        # Check if colmap output path exists. We compute stats for track
        # length, num_cameras, num 3d points, only when colmap succeeds
        if os.path.exists(colmap_res_path):
            # Read colmap models
            _, images, _ = read_model(path=colmap_res_path, ext='.bin')
            first_xyz, second_xyz = [], []
            for _, value in images.items():

                # Get ground truth translation
                t_gt = calib_dict[value.name.split('.')[0]]['T']
                r_gt = calib_dict[value.name.split('.')[0]]['R']
                first_xyz.append(-np.dot(r_gt.T, t_gt))

                # Get actual translation
                t = np.asarray(value.tvec).flatten().reshape((3, ))
                r = np.asarray(qvec2rotmat(value.qvec))
                second_xyz.append(-np.dot(r.T, t))

            first_xyz = np.asarray(first_xyz).transpose()
            second_xyz = np.asarray(second_xyz).transpose()
            num_points = first_xyz.shape[1]
            if num_points >= 3:
                prob_inlier = max(3 / num_points, 0.5)
                num_iter = int(calc_num_iter_ransac(prob_inlier))
                std = calc_std(first_xyz)
                max_trans_error = calc_max_trans_error(first_xyz)
                rot, trans, scale, trans_error, num_inlier = ate_ransac(
                    second_xyz, first_xyz, num_iter, std * 0.5)
                if trans_error > max_trans_error or \
                        num_inlier < 0.3 * num_points:
                    trans_error = max_trans_error
                ate_list += [trans_error]

    # Aggregate results for all bags
    res_dict['avg_ate'] = float(np.mean(ate_list))


def compute_num_input_matches(res_dict, deprecated_images, cfg):
    '''Save the number of input matches given to Colmap.'''

    # TODO fix this after re-implementing custom matches
    # if cfg.method_dict['config_{}_{}'.format(cfg.dataset,
    #                                          cfg.task)]['use_custom_matches']:
    #     raise NotImplementedError(
    #         'TODO Load the right dict with custom matches')

    # Read match dict
    matches_dict = load_h5_valid_image(
        get_filter_match_file(cfg) ,deprecated_images)

    # For every bag, compute the number of matches going into colmap
    bag_size_json = load_json(
        getattr(cfg, 'splits_{}_{}'.format(cfg.dataset, cfg.subset)))
    bag_size_list = [b['bag_size'] for b in bag_size_json]
    bag_size_num = [b['num_in_bag'] for b in bag_size_json]

    # Average it per bag size first, then across all bag sizes
    num_input_matches = []
    for bag_size, cur_bag_size_num in zip(bag_size_list, bag_size_num):
        num_input_matches_bagsize = []
        for bag_id in range(cur_bag_size_num):
            cfg_bag = deepcopy(cfg)
            cfg_bag.bag_size = bag_size
            cfg_bag.bag_id = bag_id

            # Skip if bag contain deprecated images
            if not valid_bag(cfg_bag, deprecated_images):
                continue

            images = get_colmap_image_path_list(cfg_bag)
            keys = [os.path.splitext(os.path.basename(im))[0] for im in images]
            pairs = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    pairs.append('-'.join(
                        sorted([keys[i], keys[j]], reverse=True)))
            for pair in pairs:
                num_input_matches_bagsize.append(matches_dict[pair].shape[-1])
        num_input_matches.append(np.mean(num_input_matches_bagsize))

    res_dict['num_input_matches'] = np.mean(num_input_matches)


def compute_colmap_stats(res_dict, deprecated_images, cfg):
    '''Retrieve stats from Colmap and add them to the dictionary.'''

    track_length_list = []
    num_cameras_list = []
    is_fail_list = []
    num_3d_pts_list = []

    # For all the bags
    cfg_bag = deepcopy(cfg)
    for bag_id in range(get_num_in_bag(cfg_bag)):
        cfg_bag.bag_id = bag_id

        # Skip if bag contain deprecated images
        if not valid_bag(cfg_bag, deprecated_images):
            continue

        # Check if colmap output path exists. We compute stats for track
        # length, num_cameras, num 3d points, only when colmap succeeds
        if not os.path.exists(get_colmap_output_path(cfg_bag)):
            # track_length = 0
            # num_cameras = 0
            is_fail_list += [1]
            # num_3d_pts = 0

        else:
            is_fail_list += [0]

            # Get colmap output (binary files) path
            colmap_res_path = os.path.join(get_colmap_output_path(cfg_bag),
                                           str(get_best_colmap_index(cfg_bag)))

            # Read colmap models
            cameras, images, points = read_model(path=colmap_res_path,
                                                 ext='.bin')
            # Track length
            num_obs = []
            for idx in points:
                num_obs.append(len(np.unique(points[idx].image_ids)))
            if len(num_obs) == 0:
                num_obs = 0
            else:
                num_obs = np.array(num_obs)
                num_obs = num_obs.mean()
            track_length_list += [num_obs]
            # Number of cameras
            num_cameras_list += [len(list(images.keys()))]
            # Number of 3D points
            num_3d_pts_list += [len(points)]

    # Aggregate results for all bags
    res_dict['avg_track_length'] = float(
        np.mean(track_length_list)) if len(track_length_list) > 0 else 0.0
    res_dict['avg_num_cameras'] = float(
        np.mean(num_cameras_list)) if len(num_cameras_list) > 0 else 0.0
    res_dict['success_rate'] = float(1.0 - np.mean(is_fail_list))
    res_dict['avg_num_3d_points'] = float(
        np.mean(num_3d_pts_list)) if len(num_3d_pts_list) > 0 else 0.0
    res_dict['num_in_bag'] = float(get_num_in_bag(cfg_bag))


def average_stereo_over_runs(cfg, res_dict, num_runs):
    '''Average results in res_dict over multiple runs and scenes. Key
    res_dict[scene]['stereo']['run_avg'] must already exist.
    '''

    for scene in res_dict:
        if scene == 'allseq':
            continue

        for metric in res_dict[scene]['stereo']['run_0']:
            values = []
            for run in range(num_runs):
                values.append(
                    res_dict[scene]['stereo']['run_{}'.format(run)][metric])

            # Compute statistics: for arrays, only the mean
            values = np.array(values)
            if values.ndim > 1:
                res_dict[scene]['stereo']['run_avg'][metric] = OrderedDict({
                    'mean':
                    np.mean(values, axis=0).tolist(),
                })
            else:
                res_dict[scene]['stereo']['run_avg'][metric] = OrderedDict({
                    'mean':
                    np.mean(values).tolist(),
                    'std_runs':
                    np.std(values).tolist(),
                    'min_runs':
                    np.min(values).tolist(),
                    'max_runs':
                    np.max(values).tolist(),
                })


def average_multiview_over_runs(cfg, res_dict, num_runs, bag_keys):
    '''Average results in res_dict over multiple runs and scenes. Key
    res_dict[scene]['multiview']['run_avg'] must already exist.
    '''

    for scene in res_dict:
        if scene == 'allseq':
            continue

        for bag in bag_keys:
            for metric in res_dict[scene]['multiview']['run_0'][bag]:
                values = []
                for run in range(num_runs):
                    values.append(res_dict[scene]['multiview']['run_{}'.format(
                        run)][bag][metric])

                # Compute statistics: for arrays, only the mean
                values = np.array(values)
                if values.ndim > 1:
                    res_dict[scene][
                        cfg.task]['run_avg'][bag][metric] = OrderedDict({
                            'mean':
                            np.nanmean(values, axis=0).tolist()
                            if metric.lower() == 'avg_ate' else np.mean(
                                values, axis=0).tolist(),
                        })
                else:
                    res_dict[scene][
                        cfg.task]['run_avg'][bag][metric] = OrderedDict({
                            'mean':
                            np.nanmean(values.tolist()) if
                            metric.lower() == 'avg_ate' else np.mean(values),
                            'std_runs':
                            np.nanstd(values.tolist())
                            if metric.lower() == 'avg_ate' else np.std(
                                values.tolist()),
                            'min_runs':
                            np.nanmin(values.tolist())
                            if metric.lower() == 'avg_ate' else np.min(
                                values.tolist()),
                            'max_runs':
                            np.nanmax(values.tolist())
                            if metric.lower() == 'avg_ate' else np.max(
                                values.tolist()),
                        })


def average_stereo_over_scenes(cfg, res_dict, num_runs):
    '''Average dictionary results over multiple scenes and runs (stereo).
    '''

    # No need to aggregate runs for allseq, just compute the statistics
    all_dict = res_dict['allseq']['stereo']['run_avg']

    scenes_minus_allseq = [s for s in res_dict if s != 'allseq']
    num_scenes = len(scenes_minus_allseq)
    for idx_scene, scene in enumerate(scenes_minus_allseq):
        if scene == 'allseq':
            continue

        for run in range(num_runs):
            run_str = 'run_{}'.format(run)
            cur_dict = res_dict[scene]['stereo'][run_str]

            # Accumulate items
            for _key, _value in cur_dict.items():
                if _key not in all_dict:
                    all_dict[_key] = np.zeros(
                        (num_runs, num_scenes,
                         len(_value) if isinstance(_value, list) else 1))
                all_dict[_key][run][idx_scene] = _value

    # Populate 'allseq'
    for _key, _value in all_dict.items():
        # Compute statistics other than the mean for scalars only
        if _value.shape[-1] > 1:
            all_dict[_key] = OrderedDict({
                'mean':
                np.mean(_value, axis=(0, 1)).tolist(),
            })
        else:
            all_dict[_key] = OrderedDict({
                'mean':
                _value.mean(),
                'std_scenes':
                _value.mean(axis=0).std(),
                'std_runs':
                _value.mean(axis=1).std(),
            })


def average_multiview_over_scenes(cfg, res_dict, num_runs, bags):
    '''Average dictionary results over multiple scenes and runs (multiview).
    '''

    # No need to aggregate runs for allseq, just compute the statistics
    all_dict = res_dict['allseq']['multiview']['run_avg']

    scenes_minus_allseq = [s for s in res_dict if s != 'allseq']
    num_scenes = len(scenes_minus_allseq)
    for idx_scene, scene in enumerate(scenes_minus_allseq):
        if scene == 'allseq':
            continue

        for run in range(num_runs):
            run_str = 'run_{}'.format(run)
            for bag_size in bags:
                cur_dict = res_dict[scene]['multiview'][run_str][bag_size]

                # Accumulate items
                for _key, _value in cur_dict.items():
                    if _key not in all_dict[bag_size]:
                        all_dict[bag_size][_key] = np.zeros(
                            (num_runs, num_scenes,
                             len(_value) if isinstance(_value, list) else 1))
                    all_dict[bag_size][_key][run][idx_scene] = _value

    # Populate 'allseq'
    for bag_size in bags:
        for _key, _value in all_dict[bag_size].items():
            # Compute statistics other than the mean for scalars only
            if _value.shape[-1] > 1:
                all_dict[bag_size][_key] = OrderedDict({
                    'mean':
                    np.mean(_value, axis=(0, 1)).tolist(),
                })
            else:
                # TODO figure out if this is still a problem for ATE
                if _key != 'avg_ate':
                    all_dict[bag_size][_key] = OrderedDict({
                        'mean':
                        _value.mean(),
                        'std_scenes':
                        _value.mean(axis=0).std(),
                        'std_runs':
                        _value.mean(axis=1).std(),
                    })
                else:
                    all_dict[bag_size][_key] = OrderedDict({
                        'mean':
                        np.nanmean(_value),
                        'std_scenes':
                        np.nanstd(np.nanmean(_value, axis=0)),
                        'std_runs':
                        np.nanstd(np.nanmean(_value, axis=1)),
                    })


def average_multiview_over_bags(cfg, res_dict, bag_size_int):
    '''Average results in res_dict over multiple bags and scenes. Key
    res_dict[scene][task]'''

    avg_dict = res_dict['bag_avg']

    # Some metrics cannot be simply averaged
    for metric in res_dict['{}bag'.format(bag_size_int[-1])]:
        values = []
        for bag_int in bag_size_int:
            bag_name = '{}bag'.format(bag_int)
            if metric in [
                    'avg_num_keypoints', 'qt_auc_colmap_05',
                    'qt_auc_colmap_10', 'avg_track_length', 'success_rate',
                    'avg_num_3d_points', 'num_input_matches'
            ]:
                values += [res_dict[bag_name][metric]]
                avg_dict[metric] = np.mean(values)
            elif metric == 'avg_ate':
                values += [res_dict[bag_name][metric]]
                avg_dict[metric] = np.nanmean(values)
            elif metric == 'avg_num_cameras':
                values += [res_dict[bag_name][metric] / bag_int]
                avg_dict['ratio_registered_cameras'] = np.mean(values)
            elif metric == 'qt_colmap_01_10':
                values += [res_dict[bag_name][metric]]
                avg_dict[metric] = np.mean(np.asarray(values), axis=0).tolist()
            elif metric in ['num_in_bag']:
                pass
            else:
                raise RuntimeError(
                    'Not sure how to average "{}"'.format(metric))


def get_descriptor_properties(cfg, descriptors_dict):
    '''Store descriptor size and type in the results file.'''

    descriptor_type, descriptor_size, descriptor_nbytes = None, None, None
    for item in descriptors_dict.values():
        descriptor_type = str(item.dtype)
        descriptor_size = item.shape[-1]
        descriptor_nbytes = item[0].nbytes
        break

    if not descriptor_type or not descriptor_size or not descriptor_nbytes:
        raise ValueError('Cannot retrieve descriptor properties')

    return descriptor_type, descriptor_size, descriptor_nbytes
