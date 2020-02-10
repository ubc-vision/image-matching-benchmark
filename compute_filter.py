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

from __future__ import print_function
import sys
import os
import pickle
from tqdm import tqdm
import numpy as np
from shutil import copytree, rmtree, copyfile
import cv2
import subprocess
import argparse
from time import time

from config import get_config, print_usage
from utils.io_helper import load_h5, save_h5
from utils.load_helper import load_calib
from utils.path_helper import (get_data_path, get_fullpath_list,
                               get_match_file, get_kp_file, get_cne_temp_path,
                               get_cne_data_dump_path, get_filter_match_file,
                               get_filter_path, get_filter_cost_file)
from utils.queue_helper import create_sh_cmd


def make_xy(sfm_cfg):
    """
    Messy conveniency function to re-format our data into that expected by the
    Context Networks release.
    """

    xs = []
    ys = []
    Rs = []
    ts = []
    cx1s = []
    cy1s = []
    f1s = []
    cx2s = []
    cy2s = []
    f2s = []
    key_list = []

    data_dir = get_data_path(sfm_cfg)

    keypoints_dict = load_h5(get_kp_file(sfm_cfg))
    match_dict = load_h5(get_match_file(sfm_cfg))
    calib_list = get_fullpath_list(data_dir, 'calibration')
    calib_dict = load_calib(calib_list)

    print('Converting data to a CNe friendly format...')
    for image_pair, match_idx_pairs in tqdm(match_dict.items()):
        key_list.append(image_pair)

        # Get image name and read image
        image_1, image_2 = image_pair.split('-')
        image1 = cv2.imread(os.path.join(data_dir, 'images', image_1 + '.jpg'))
        image2 = cv2.imread(os.path.join(data_dir, 'images', image_2 + '.jpg'))

        # Get dR
        R_1 = calib_dict[image_1]['R']
        R_2 = calib_dict[image_2]['R']
        dR = np.dot(R_2, R_1.T)

        # Get dt
        t_1 = calib_dict[image_1]['T'].reshape((3, 1))
        t_2 = calib_dict[image_2]['T'].reshape((3, 1))
        dt = t_2 - np.dot(dR, t_1)

        # Save R, t for evaluation
        Rs += [np.array(dR).reshape(3, 3)]

        # normalize t before saving
        dtnorm = np.sqrt(np.sum(dt**2))
        assert (dtnorm > 1e-5)
        dt /= dtnorm
        ts += [np.array(dt).flatten()]

        # Save img1, center offset, f
        # img1s += [image1.transpose(2, 0, 1)]
        cx1 = (image1.shape[1] - 1.0) * 0.5
        cy1 = (image1.shape[0] - 1.0) * 0.5
        f1 = max(image1.shape[1] - 1.0, image1.shape[0] - 1.0)
        cx1s += [cx1]
        cy1s += [cy1]
        f1s += [f1]

        # Save img2, center offset, f
        # img2s += [image2.transpose(2, 0, 1)]
        cx2 = (image2.shape[1] - 1.0) * 0.5
        cy2 = (image2.shape[0] - 1.0) * 0.5
        f2 = max(image2.shape[1] - 1.0, image2.shape[0] - 1.0)
        cx2s += [cx2]
        cy2s += [cy2]
        f2s += [f2]

        # Get key points
        kp1 = np.asarray(keypoints_dict[image_1])
        kp1 = kp1[:, :2]
        kp2 = np.asarray(keypoints_dict[image_2])
        kp2 = kp2[:, :2]

        # Normalize Key points
        kp1 = (kp1 - np.asarray([cx1, cy1]).T) / np.asarray([f1, f1]).T
        kp2 = (kp2 - np.asarray([cx2, cy2]).T) / np.asarray([f2, f2]).T

        # Shuffle key points based on match index
        x1_index = match_idx_pairs[0, :]
        x2_index = match_idx_pairs[1, :]

        # Get shuffled key points for image 1
        x1 = kp1[x1_index, :]

        # Assume depth = 1
        z = np.ones((x1.shape[0], 1))

        # Construct 3D points
        y1 = np.concatenate([x1 * z, z], axis=1)

        # Project 3D points to image 2
        y1p = np.matmul(dR[None], y1[..., None]) + dt[None]

        # move back to the canonical plane
        x1p = y1p[:, :2, 0] / y1p[:, 2, 0][..., None]

        # Get shuffled key points for image 2
        x2 = kp2[x2_index, :]

        # make xs in NHWC
        xs += [
            np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose(
                (1, 2, 0))
        ]
        # Get the geodesic distance using with x1, x2, dR, dt
        geod_d = get_sampsons(x1, x2, dR, dt)

        # Get *rough* reprojection errors. Note that the depth may be noisy. We
        # ended up not using this...
        reproj_d = np.sum((x2 - x1p)**2, axis=1)

        # add to label list
        ys += [np.stack([geod_d, reproj_d], axis=1)]

    res_dict = {}
    res_dict['xs'] = xs
    res_dict['ys'] = ys
    res_dict['Rs'] = Rs
    res_dict['ts'] = ts
    res_dict['cx1s'] = cx1s
    res_dict['cy1s'] = cy1s
    res_dict['f1s'] = f1s
    res_dict['cx2s'] = cx2s
    res_dict['cy2s'] = cy2s
    res_dict['f2s'] = f2s

    return res_dict, key_list


def save_match_inlier(sfm_cfg, key_list, mask_dict):
    match_dict = load_h5(get_match_file(sfm_cfg))

    if len(match_dict) != len(mask_dict):
        raise RuntimeError('Number of pairs from CNe output is different '
                           'from original data!')

    for key, match_mask in mask_dict.items():
        mask_index = np.where(match_mask)
        match_idx_pairs_inlier = match_dict[key_list[key]][:, mask_index]
        match_dict[key_list[key]] = np.squeeze(match_idx_pairs_inlier)

    save_h5(match_dict, get_filter_match_file(sfm_cfg))


def get_cne_config():
    sys.argv = []
    cne_cfg, unparsed = get_cne_config_from_cne()
    cne_cfg.run_mode = 'test'
    cne_cfg.log_dir = 'models/nd_bp'
    cne_cfg.res_dir = 'third_party/cne'
    cne_cfg.data_crop_center = False
    return cne_cfg


def cne_interface(sfm_cfg):
    """Entry point to refine matches with CNe.
    
    Parameters
    ----------
    sfm_cfg: Config.
    """

    # Get data
    data, key_list = make_xy(sfm_cfg)
    data_dict = {}
    data_dict['test'] = data

    # Construct cne config
    cne_cfg = get_cne_config()

    # Init network
    mynet = MyNetwork(cne_cfg)

    # Run CNe
    t_start = time()
    mask_dict = mynet.test(data_dict)
    ellapsed = time() - t_start

    # Save CNe timings
    save_h5({'cost': ellapsed / len(key_list)}, get_filter_cost_file(sfm_cfg))
    print('CNe cost (averaged over image pairs): {:0.2f} sec'.format(
        np.mean(ellapsed / len(key_list))))

    # Extract match mask
    save_match_inlier(sfm_cfg, key_list, mask_dict)


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    if not os.path.exists(get_filter_path(cfg)):
        os.makedirs(get_filter_path(cfg))

    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    if cur_key not in cfg.method_dict:
        raise ValueError('Cannot find "{}"'.format(cur_key))
    cur_filter = cfg.method_dict[cur_key]['outlier_filter']

    if cur_filter['method'] == 'cne-bp-nd':
        from third_party.cne.config import get_config as get_cne_config_from_cne
        from third_party.cne.network import MyNetwork
        from third_party.cne.geom import get_sampsons
        cne_interface(cfg)
    elif cur_filter['method'] == 'none':
        copyfile(get_match_file(cfg), get_filter_match_file(cfg))
        save_h5({'cost': 0.0}, get_filter_cost_file(cfg))
    else:
        raise ValueError('Unknown prefilter type')
