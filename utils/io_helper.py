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

import json
from jsmin import jsmin
import cv2
import h5py
import numpy as np


def build_composite_image(image_path1,
                          image_path2,
                          axis=1,
                          margin=0,
                          background=1):
    '''
    Load two images and returns a composite image.

    Parameters
    ----------
    image_path1: Fullpath to image 1.
    image_path2: Fullpath to image 2.
    in: Space between images
    ite)

    Returns
    -------
    (Composite image,
        (vertical_offset1, vertical_offset2),
        (horizontal_offset1, horizontal_offset2))
    '''

    if background != 0 and background != 1:
        background = 1
    if axis != 0 and axis != 1:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal')

    im1 = cv2.imread(image_path1)
    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    elif im1.ndim == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError('invalid image format')

    im2 = cv2.imread(image_path2)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    elif im2.ndim == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError('invalid image format')

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3),
                             dtype=np.uint8) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3),
                             dtype=np.uint8) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))


def load_json(json_path):
    '''Loads JSON file.'''

    with open(json_path) as js_file:
        out = parse_json(js_file.read())
    return out


def parse_json(str1):
    minified = jsmin(str1).replace('\n', ' ')
    minified = minified.replace(',]', ']')
    minified = minified.replace(',}', '}')
    if minified.startswith('"') and minified.endswith('"'):
        minified = minified[1:-1]

    json_load = json.loads(minified)
    return json_load


def save_h5(dict_to_save, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def load_h5(filename):
    '''Loads dictionary from hdf5 file'''

    dict_to_load = {}
    try:
        with h5py.File(filename, 'r') as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key].value
    except:
        print('Cannot find file {}'.format(filename))
    return dict_to_load
