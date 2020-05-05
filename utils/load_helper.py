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
import cv2
import numpy as np
from utils.io_helper import load_h5


def load_image(image_path,
               use_color_image=False,
               input_width=512,
               crop_center=True,
               force_rgb=False):
    '''
    Loads image and do preprocessing.

    Parameters
    ----------
    image_path: Fullpath to the image.
    use_color_image: Flag to read color/gray image
    input_width: Width of the image for scaling
    crop_center: Flag to crop while scaling
    force_rgb: Flag to convert color image from BGR to RGB

    Returns
    -------
    Tuple of (Color/Gray image, scale_factor)
    '''

    # Assuming all images in the directory are color images
    image = cv2.imread(image_path)
    if not use_color_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif force_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop center and resize image into something reasonable
    scale_factor = 1.0
    if crop_center:
        rows, cols = image.shape[:2]
        if rows > cols:
            cut = (rows - cols) // 2
            img_cropped = image[cut:cut + cols, :]
        else:
            cut = (cols - rows) // 2
            img_cropped = image[:, cut:cut + rows]
        scale_factor = float(input_width) / float(img_cropped.shape[0])
        image = cv2.resize(img_cropped, (input_width, input_width))

    return (image, scale_factor)


def load_depth(depth_path):
    return load_h5(depth_path)['depth']


def load_vis(vis_fullpath_list, subset_index=None):
    '''
    Given fullpath_list load all visibility ranges
    '''
    vis = []
    if subset_index is None:
        for vis_file in vis_fullpath_list:
            # Load visibility
            vis.append(np.loadtxt(vis_file).flatten().astype('float32'))
    else:
        for idx in subset_index:
            tmp_vis = np.loadtxt(
                vis_fullpath_list[idx]).flatten().astype('float32')
            tmp_vis = tmp_vis[subset_index]
            vis.append(tmp_vis)
    return vis


def load_calib(calib_fullpath_list, subset_index=None):
    '''Load all calibration files and create a dictionary.'''

    calib = {}
    if subset_index is None:
        for _calib_file in calib_fullpath_list:
            img_name = os.path.splitext(
                os.path.basename(_calib_file))[0].replace('calibration_', '')
            # _calib_file.split(
            #     '/')[-1].replace('calibration_', '')[:-3]
            # # Don't know why, but rstrip .h5 also strips
            # # more then necssary sometimes!
            # #
            # # img_name = _calib_file.split(
            # #     '/')[-1].replace('calibration_', '').rstrip('.h5')
            calib[img_name] = load_h5(_calib_file)
    else:
        for idx in subset_index:
            _calib_file = calib_fullpath_list[idx]
            img_name = os.path.splitext(
                os.path.basename(_calib_file))[0].replace('calibration_', '')
            calib[img_name] = load_h5(_calib_file)
    return calib

def load_h5_valid_image(path, deprecated_images):
    return remove_keys(load_h5(path),deprecated_images)

def remove_keys(d, key_list):
    for key in key_list:
        del_key_list = [tmp_key for tmp_key in d.keys() if key in tmp_key]
        for del_key in del_key_list:
            del d[del_key]
    return d