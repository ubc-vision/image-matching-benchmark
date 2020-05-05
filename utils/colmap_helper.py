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
import numpy as np

from third_party.colmap.scripts.python.read_write_model import (qvec2rotmat,
                                                         read_images_binary)
from utils.eval_helper import evaluate_R_t
from utils.path_helper import (get_colmap_mark_file, get_colmap_output_path,
                               get_colmap_pose_file, get_colmap_temp_path,
                               get_data_path, get_fullpath_list,
                               parse_file_to_list, get_item_name_list)

def valid_bag(cfg_bag, deprecated_images):
    # Get list of images
    image_path_list = get_colmap_image_path_list(cfg_bag)
    image_name_list = get_item_name_list(image_path_list)
    # skip if bag contain deprecated image
    if len(list(set(image_name_list) & set(deprecated_images)))!=0:
        return False
    return True 

def is_colmap_complete(cfg):
    '''Checks if stereo evaluation is complete.'''

    # We should have the colmap pose file and no colmap temp path
    is_complete = os.path.exists(get_colmap_pose_file(cfg)) and (
        not os.path.exists(get_colmap_temp_path(cfg)))

    return is_complete


def is_colmap_img_valid(colmap_img_file):
    '''Return validity of a colmap reconstruction'''

    images_bin = read_images_binary(colmap_img_file)
    # Check if everything is finite for this subset
    for key in images_bin.keys():
        q = np.asarray(images_bin[key].qvec).flatten()
        t = np.asarray(images_bin[key].tvec).flatten()

        is_cur_valid = True
        is_cur_valid = is_cur_valid and q.shape == (4, )
        is_cur_valid = is_cur_valid and t.shape == (3, )
        is_cur_valid = is_cur_valid and np.all(np.isfinite(q))
        is_cur_valid = is_cur_valid and np.all(np.isfinite(t))

        # If any is invalid, immediately return
        if not is_cur_valid:
            return False

    return True


def has_colmap_run(cfg):
    '''Checks if stereo evaluation is complete.'''

    is_complete = os.path.exists(get_colmap_mark_file(cfg))

    return is_complete


def get_colmap_image_path_list(cfg):
    '''Gives a list of all images in this bag.'''
    data_dir = get_data_path(cfg)
    list_file = os.path.join(
        data_dir, 'sub_set', '{}bag_{:03d}.txt'.format(cfg.bag_size,
                                                       cfg.bag_id))

    image_path_list = parse_file_to_list(list_file, data_dir)

    return image_path_list


def get_colmap_image_subset_index(cfg, image_subset_list):
    data_dir = get_data_path(cfg)
    images_list = get_fullpath_list(data_dir, 'images')
    subset_index = []
    for image2 in image_subset_list:
        for i, image1 in enumerate(images_list):
            if image2 == image1:
                subset_index.append(i)
    return subset_index


def get_colmap_vis_list(cfg):
    '''Gives a list of visibilities in this bag.'''

    data_dir = get_data_path(cfg).rstrip('/') + '_{}bag_{:03d}'.format(
        cfg.bag_size, cfg.bag_id)
    vis_list = get_fullpath_list(data_dir, 'visibility')

    return vis_list


def get_colmap_calib_list(cfg):
    '''
    Gives a list of calibration files in this bag.
    '''
    data_dir = get_data_path(cfg)
    calib_list = get_fullpath_list(data_dir, 'calibration')

    return calib_list


def get_best_colmap_index(cfg):
    '''
    Determines the colmap model with the most images if there is more than one.
    '''

    colmap_output_path = get_colmap_output_path(cfg)

    # First find the colmap reconstruction with the most number of images.
    best_index, best_num_images = -1, 0

    # Check all valid sub reconstructions.
    if os.path.exists(colmap_output_path):
        idx_list = [
            _d for _d in os.listdir(colmap_output_path)
            if os.path.isdir(os.path.join(colmap_output_path, _d))
        ]
    else:
        idx_list = []

    for cur_index in idx_list:
        cur_output_path = os.path.join(colmap_output_path, cur_index)
        if os.path.isdir(cur_output_path):
            colmap_img_file = os.path.join(cur_output_path, 'images.bin')
            images_bin = read_images_binary(colmap_img_file)
            # Check validity
            if not is_colmap_img_valid(colmap_img_file):
                continue
            # Find the reconstruction with most number of images
            if len(images_bin) > best_num_images:
                best_index = int(cur_index)
                best_num_images = len(images_bin)

    return best_index


def compute_stereo_metrics_from_colmap(img1, img2, calib1, calib2, best_index,
                                       cfg):
    '''Computes (pairwise) error metrics from Colmap results.'''

    # Load COLMAP dR and dt
    colmap_output_path = get_colmap_output_path(cfg)

    # First read images.bin for the best reconstruction
    images_bin = read_images_binary(
        os.path.join(colmap_output_path, str(best_index), 'images.bin'))

    # For each key check if images_bin[key].name = image_name
    R_1_actual, t_1_actual = None, None
    R_2_actual, t_2_actual = None, None
    for key in images_bin.keys():
        if images_bin[key].name == os.path.basename(img1):
            R_1_actual = qvec2rotmat(images_bin[key].qvec)
            t_1_actual = images_bin[key].tvec
        if images_bin[key].name == os.path.basename(img2):
            R_2_actual = qvec2rotmat(images_bin[key].qvec)
            t_2_actual = images_bin[key].tvec

    # Compute err_q and err_t only when R, t are not None
    err_q, err_t = np.inf, np.inf
    if (R_1_actual is not None) and (R_2_actual is not None) and (
            t_1_actual is not None) and (t_2_actual is not None):
        # Compute dR, dt (actual)
        dR_act = np.dot(R_2_actual, R_1_actual.T)
        dt_act = t_2_actual - np.dot(dR_act, t_1_actual)

        # Get R, t from calibration information
        R_1, t_1 = calib1['R'], calib1['T'].reshape((3, 1))
        R_2, t_2 = calib2['R'], calib2['T'].reshape((3, 1))

        # Compute ground truth dR, dt
        dR = np.dot(R_2, R_1.T)
        dt = t_2 - np.dot(dR, t_1)

        # Save err_, err_t
        err_q, err_t = evaluate_R_t(dR, dt, dR_act, dt_act)

    return err_q, err_t
