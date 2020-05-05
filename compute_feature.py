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
from tqdm import tqdm
from config import get_config
from methods import local_feature as lfeat
from utils.io_helper import save_h5
from utils.path_helper import (get_data_path, get_desc_file, get_feature_path,
                               get_fullpath_list, get_item_name_list,
                               get_kp_file, get_angle_file, get_scale_file,
                               get_affine_file, get_score_file)
import cv2


def compute_per_img_file(img_path, cfg):
    '''Computes features and returns them.

    Parameters
    ----------
    img_path: str
        Path to the image file to work on

    cfg: Namespace
        Configuration arguments

    Returns
    -------
    keypoints: list
        List of keypoints
    descriptors: list
        List of descriptors
    '''

    if cfg.num_opencv_threads > 0:
        cv2.setNumThreads(cfg.num_opencv_threads)

    # Check if we know this keypoint detector
    kp = cfg.method_dict['config_common']['keypoint'].lower()
    desc = cfg.method_dict['config_common']['descriptor'].lower()

    # SIFT and root-SIFT, with CLAHE
    if kp in [
            u + v for u in ['sift-def', 'sift-lowth'] for v in ['', '-clahe']
    ]:
        if desc in [
                u + v + w for u in ['sift', 'rootsift']
                for v in ['', '-clahe'] for w in ['', '-upright', '-upright--']
        ]:
            return lfeat.sift.run(img_path, cfg)

    # ORB
    if kp == 'orb' and desc == 'orb':
        return lfeat.orb.run(img_path, cfg)

    # SURF
    if kp in ['surf-def', 'surf-lowth'] and desc == 'surf':
        return lfeat.surf.run(img_path, cfg)

    # AKAZE
    if kp in ['akaze-def', 'akaze-lowth'] and desc == 'akaze':
        return lfeat.akaze.run(img_path, cfg)

    # FREAK
    if kp in ['freak-def', 'freak-lowth'] and desc == 'freak':
        return lfeat.freak.run(img_path, cfg)

    # Preserving this for now
    if kp == 'sift8k' and desc == 'affnethardnetextract':
        return lfeat.sift8k_affnethardnetextract.run(img_path, cfg)
    if kp == 'sift8k' and desc == 'hardnetextract':
        return lfeat.sift8k_hardnetextract.run(img_path, cfg)

    raise RuntimeError('Unknown keypoint/descriptor combination')


def main(cfg):
    '''Main function to compute features.

    Parameters
    ----------
    cfg: Namespace
        Configuration
    '''

    if os.path.exists(get_kp_file(cfg)) and os.path.exists(get_desc_file(cfg)):
        print(' -- already exists, skipping feature extraction')
        return

    # Get data directory
    data_dir = get_data_path(cfg)

    # Get list of all images and visibility files in the 'set_100'
    images_list = get_fullpath_list(data_dir, 'images')

    # Also create a list which only contains the image names, so that it can be
    # used as keys in the dictionary later
    image_names = get_item_name_list(images_list)

    # Create folder
    save_dir = get_feature_path(cfg)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Compute and save keypoints and descriptors
    #
    # Parallel processing actually slows down stuff, because opencv is already
    # using multiple threads. We just simply go through one by one without
    # parallel processing for now
    print('Extracting Keypoints and Descriptors:')
    result = []
    for img_path in tqdm(images_list):
        result.append(compute_per_img_file(img_path, cfg))

    # num_cores = int(multiprocessing.cpu_count() * 0.9)
    # print('Extracting Keypoints and Descriptors:')
    # result = Parallel(n_jobs=num_cores)(delayed(compute_per_img_file)(
    #     img_path, cfg) for img_path in tqdm(images_list))

    # Save keypoints and descriptors
    kp_dict = {}
    scale_dict = {}
    angle_dict = {}
    score_dict = {}
    descs_dict = {}
    affine_dict = {}
    for _i in range(len(image_names)):
        assert 'kp' in result[_i], 'Must provide keypoints'
        assert 'descs' in result[_i], 'Must provide descriptors'
        if 'kp' in result[_i]:
            kp_dict[image_names[_i]] = result[_i]['kp']
        if 'scale' in result[_i]:
            scale_dict[image_names[_i]] = result[_i]['scale']
        if 'angle' in result[_i]:
            angle_dict[image_names[_i]] = result[_i]['angle']
        if 'affine' in result[_i]:
            affine_dict[image_names[_i]] = result[_i]['affine']
        if 'score' in result[_i]:
            score_dict[image_names[_i]] = result[_i]['score']
        if 'descs' in result[_i]:
            descs_dict[image_names[_i]] = result[_i]['descs']

    # Finally, save packed keypoints and descriptors
    save_h5(kp_dict, get_kp_file(cfg))
    save_h5(scale_dict, get_scale_file(cfg))
    save_h5(angle_dict, get_angle_file(cfg))
    save_h5(score_dict, get_score_file(cfg))
    save_h5(descs_dict, get_desc_file(cfg))
    save_h5(affine_dict, get_affine_file(cfg))


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print(unparsed)
        print_usage()
        exit(1)

    main(cfg)
