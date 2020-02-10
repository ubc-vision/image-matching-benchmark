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
import subprocess
from shutil import copyfile, rmtree

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from config import get_config, print_usage
from utils.colmap_helper import (compute_stereo_metrics_from_colmap,
                                 get_best_colmap_index,
                                 get_colmap_image_path_list,
                                 is_colmap_img_valid,
                                 get_colmap_image_subset_index)
from utils.io_helper import load_h5, save_h5
from utils.load_helper import load_calib
from utils.match_helper import compute_image_pairs
from utils.path_helper import (get_colmap_output_path, get_colmap_pose_file,
                               get_colmap_temp_path, get_item_name_list,
                               get_kp_file, get_match_file, get_fullpath_list,
                               get_data_path, get_filter_match_file)


def compute_pose_error(cfg):
    '''
    Computes the error using quaternions and translation vector for COLMAP
    '''

    if os.path.exists(get_colmap_pose_file(cfg)):
        print(' -- already exists, skipping COLMAP eval')
        return

    # Load visiblity and images
    image_path_list = get_colmap_image_path_list(cfg)
    subset_index = get_colmap_image_subset_index(cfg, image_path_list)
    image_name_list = get_item_name_list(image_path_list)

    # Load camera information
    data_dir = get_data_path(cfg)
    calib_list = get_fullpath_list(data_dir, 'calibration')
    calib_dict = load_calib(calib_list, subset_index)

    # Generate all possible pairs from all images
    pair_list = []
    for ii in range(len(image_path_list)):
        for jj in range(ii + 1, len(image_path_list)):
            pair_list.append([ii, jj])

    # Check if colmap results exist. Otherwise, this whole bag is a fail.
    colmap_output_path = get_colmap_output_path(cfg)
    is_colmap_valid = os.path.exists(os.path.join(colmap_output_path, '0'))

    if is_colmap_valid:

        # Find the best colmap reconstruction
        best_index = get_best_colmap_index(cfg)

        print('Computing pose errors')
        #num_cores = int(multiprocessing.cpu_count() * 0.9)
        num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
        result = Parallel(n_jobs=num_cores)(
            delayed(compute_stereo_metrics_from_colmap)(image_path_list[
                pair[0]], image_path_list[pair[1]], calib_dict[image_name_list[
                    pair[0]]], calib_dict[image_name_list[pair[1]]],
                                                        best_index, cfg)
            for pair in tqdm(pair_list))

    # Collect err_q, err_t from results
    err_dict = {}
    for _i in range(len(pair_list)):
        pair = pair_list[_i]
        if is_colmap_valid:
            err_q = result[_i][0]
            err_t = result[_i][1]
        else:
            err_q = np.inf
            err_t = np.inf
        err_dict[image_name_list[pair[0]] + '-' +
                 image_name_list[pair[1]]] = [err_q, err_t]

    # Finally, save packed errors
    save_h5(err_dict, get_colmap_pose_file(cfg))


def run_colmap_for_bag(cfg):
    '''Runs colmap to retrieve poses for each bag'''

    # Colmap pose file already exists, skip the session
    if os.path.exists(get_colmap_pose_file(cfg)):
        print(' -- already exists, skipping COLMAP eval')
        return

    # Load keypoints and matches
    keypoints_dict = load_h5(get_kp_file(cfg))

    matches_dict = load_h5(get_filter_match_file(cfg))

    print('Running COLMAP on bagsize {} -- bag {}'.format(
        cfg.scene, cfg.bag_size, cfg.bag_id))

    # Additional sanity check to account for crash -- in this case colmap temp
    # directory can exist. This in an indication that you need to remove
    # results and rerun colmap.
    colmap_temp_path = get_colmap_temp_path(cfg)
    colmap_output_path = get_colmap_output_path(cfg)
    if os.path.exists(colmap_temp_path):
        print(' -- temp path exists - cleaning up from crash')
        rmtree(colmap_temp_path)
        if os.path.exists(colmap_output_path):
            rmtree(colmap_output_path)
        if os.path.exists(get_colmap_pose_file(cfg)):
            os.remove(get_colmap_pose_file(cfg))

    # Check existance of colmap result and terminate if already exists.
    colmap_output_path = get_colmap_output_path(cfg)
    if os.path.exists(colmap_output_path):
        print(' -- already exists, skipping COLMAP session')
        return

    # Create output directory
    os.makedirs(colmap_output_path)

    # Create colmap temporary directory and copy files over. Remove anything
    # that might have existed.
    colmap_temp_path = get_colmap_temp_path(cfg)
    if os.path.exists(colmap_temp_path):
        rmtree(colmap_temp_path)

    # Make sure old data is gone and create a new temp folder
    assert (not os.path.exists(colmap_temp_path))
    os.makedirs(colmap_temp_path)

    # Create colmap-friendy structures
    os.makedirs(os.path.join(colmap_temp_path, 'images'))
    os.makedirs(os.path.join(colmap_temp_path, 'features'))

    # Get list of all images in this bag
    image_subset_list = get_colmap_image_path_list(cfg)

    subset_index = get_colmap_image_subset_index(cfg, image_subset_list)

    # Copy images
    for _src in image_subset_list:
        _dst = os.path.join(colmap_temp_path, 'images', os.path.basename(_src))
        copyfile(_src, _dst)

    # Write features to colmap friendly format
    for image_path in image_subset_list:
        # Retrieve image name, with and without extension
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        # Read keypoint
        keypoints = keypoints_dict[image_name_no_ext]
        # Keypoint file to write to
        kp_file = os.path.join(colmap_temp_path, 'features',
                               image_name + '.txt')
        # Open a file to write
        with open(kp_file, 'w') as f:
            # Retieve the number of keypoints
            len_keypoints = len(keypoints)
            f.write(str(len_keypoints) + ' ' + str(128) + '\n')
            for i in range(len_keypoints):
                kp = ' '.join(str(k) for k in keypoints[i][:4])
                desc = ' '.join(str(0) for d in range(128))
                f.write(kp + ' ' + desc + '\n')

    # Write matches to colmap friendly format
    # Read visibilties
    data_dir = get_data_path(cfg)
    vis_list = get_fullpath_list(data_dir, 'visibility')

    # Load matches and store them to a text file
    # TODO: This seems to be done multiple times. Do we need to do this?
    print('Generate list of all possible pairs')
    pairs = compute_image_pairs(vis_list, len(image_subset_list), cfg.vis_th,
                                subset_index)
    print('{} pairs generated'.format(len(pairs)))

    # Write to match file
    match_file = os.path.join(colmap_temp_path, 'matches.txt')
    with open(match_file, 'w') as f:
        for pair in pairs:
            image_1_name = os.path.basename(image_subset_list[pair[0]])
            image_2_name = os.path.basename(image_subset_list[pair[1]])
            image_1_name_no_ext = os.path.splitext(image_1_name)[0]
            image_2_name_no_ext = os.path.splitext(image_2_name)[0]

            # Load matches
            matches = np.squeeze(matches_dict[image_1_name_no_ext + '-' +
                                              image_2_name_no_ext])
            # only write when matches are given
            if matches.ndim == 2:
                f.write(image_1_name + ' ' + image_2_name + '\n')
                for _i in range(matches.shape[1]):
                    f.write(
                        str(matches[0, _i]) + ' ' + str(matches[1, _i]) + '\n')
                f.write('\n')
    f.close()

    # COLMAP runs -- wrapped in try except to throw errors if subprocess fails
    # and then clean up the colmap temp directory

    try:
        print('COLMAP Feature Import')
        cmd = ['colmap', 'feature_importer']
        cmd += [
            '--database_path',
            os.path.join(colmap_output_path, 'databases.db')
        ]
        cmd += ['--image_path', os.path.join(colmap_temp_path, 'images')]
        cmd += ['--import_path', os.path.join(colmap_temp_path, 'features')]

        colmap_res = subprocess.run(cmd)
        if colmap_res.returncode != 0:
            raise RuntimeError(' -- COLMAP failed to import features!')

        print('COLMAP Match Import')
        cmd = ['colmap', 'matches_importer']
        cmd += [
            '--database_path',
            os.path.join(colmap_output_path, 'databases.db')
        ]
        cmd += [
            '--match_list_path',
            os.path.join(colmap_temp_path, 'matches.txt')
        ]
        cmd += ['--match_type', 'raw']
        cmd += ['--SiftMatching.use_gpu', '0']

        colmap_res = subprocess.run(cmd)
        if colmap_res.returncode != 0:
            raise RuntimeError(' -- COLMAP failed to import matches!')

        print('COLMAP Mapper')
        cmd = ['colmap', 'mapper']
        cmd += ['--image_path', os.path.join(colmap_temp_path, 'images')]
        cmd += [
            '--database_path',
            os.path.join(colmap_output_path, 'databases.db')
        ]
        cmd += ['--output_path', colmap_output_path]
        cmd += ['--Mapper.min_model_size', str(cfg.colmap_min_model_size)]
        colmap_res = subprocess.run(cmd)
        if colmap_res.returncode != 0:
            raise RuntimeError(' -- COLMAP failed to run mapper!')

        # Delete temp directory after working
        rmtree(colmap_temp_path)

    except Exception as err:
        # Remove colmap output path and temp path
        rmtree(colmap_temp_path)
        rmtree(colmap_output_path)

        # Re-throw error
        print(err)
        raise RuntimeError('Parts of colmap runs returns failed state!')

    print('Checking validity of the colmap run just in case')

    # Check validity of colmap reconstruction for all of them
    is_any_colmap_valid = False
    idx_list = [
        os.path.join(colmap_output_path, _d)
        for _d in os.listdir(colmap_output_path)
        if os.path.isdir(os.path.join(colmap_output_path, _d))
    ]
    for idx in idx_list:
        colmap_img_file = os.path.join(idx, 'images.bin')
        if is_colmap_img_valid(colmap_img_file):
            is_any_colmap_valid = True
            break
    if not is_any_colmap_valid:
        print('Error in reading colmap output -- '
              'removing colmap output directory')
        rmtree(colmap_output_path)


def main(cfg):
    '''Main function to compute matches.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    '''

    # Run colmap
    # TODO: would be nice to run this twice if there are errors
    run_colmap_for_bag(cfg)

    # Evaluate its results
    compute_pose_error(cfg)


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(cfg)
