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

import cv2

from utils.feature_helper import convert_opencv_kp_desc
from utils.load_helper import load_image


def run(img_path, cfg):
    '''Wrapper over OpenCV ORB.

    Parameters
    ----------
    img_path (str): Path to images. 
    cfg: (Namespace): Configuration.
    '''

    # Init opencv feature extractor
    feature = cv2.ORB_create(
        nfeatures=cfg.method_dict['config_common']['num_keypoints'])

    # Load Image
    img, _ = load_image(img_path, use_color_image=False, crop_center=False)

    # Compute features
    kp, desc = feature.detectAndCompute(img, None)

    # Convert opencv keypoints into our benchmark format
    kp, desc = convert_opencv_kp_desc(
        kp, desc, cfg.method_dict['config_common']['num_keypoints'])

    result = {}
    result['kp'] = [p[0:2] for p in kp]
    result['descs'] = desc

    return result
