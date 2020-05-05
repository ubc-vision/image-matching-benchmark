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
import numpy as np

from utils.feature_helper import convert_opencv_kp_desc, l_clahe
from utils.load_helper import load_image


def run(img_path, cfg):
    '''Wrapper over OpenCV SIFT.

    Parameters
    ----------
    img_path (str): Path to images. 
    cfg: (Namespace): Configuration.

    Valid keypoint methods: "sift-def" (standard detection threshold)
    and "sift-lowth" (lowered detection threshold to extract 8000 features).
    Optional suffixes: "-clahe" (applies CLAHE over the image).

    Valid descriptors methods: "sift" and "rootsift".
    Optional suffixes: "-clahe" (applies CLAHE over the image), "upright"
    (sets keypoint orientations to 0, removing duplicates).
    '''

    # Parse options
    kp_name = cfg.method_dict['config_common']['keypoint'].lower()
    desc_name = cfg.method_dict['config_common']['descriptor'].lower()
    num_kp = cfg.method_dict['config_common']['num_keypoints']

    # Do a strict name check to prevent mistakes (e.g. due to flag order)
    if kp_name == 'sift-def':
        use_lower_det_th = False
        use_clahe_det = False
    elif kp_name == 'sift-lowth':
        use_lower_det_th = True
        use_clahe_det = False
    elif kp_name == 'sift-def-clahe':
        use_lower_det_th = False
        use_clahe_det = True
    elif kp_name == 'sift-lowth-clahe':
        use_lower_det_th = True
        use_clahe_det = True
    else:
        raise ValueError('Unknown detector')

    if desc_name == 'sift':
        use_rootsift = False
        use_clahe_desc = False
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'rootsift':
        use_rootsift = True
        use_clahe_desc = False
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'sift-clahe':
        use_rootsift = False
        use_clahe_desc = True
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'rootsift-clahe':
        use_rootsift = True
        use_clahe_desc = True
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'sift-upright':
        use_rootsift = False
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'sift-upright--':
        use_rootsift = False
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = True
    elif desc_name == 'rootsift-upright':
        use_rootsift = True
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'rootsift-upright--':
        use_rootsift = True
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = True
    elif desc_name == 'sift-clahe-upright':
        use_rootsift = False
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'sift-clahe-upright--':
        use_rootsift = False
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = True
    elif desc_name == 'rootsift-clahe-upright':
        use_rootsift = True
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'rootsift-clahe-upright--':
        use_rootsift = True
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = True
    else:
        raise ValueError('Unknown descriptor')

    # print('Extracting SIFT features with'
    #         ' use_lower_det_th={},'.format(use_lower_det_th),
    #         ' use_clahe_det={},'.format(use_clahe_det),
    #         ' use_rootsift={},'.format(use_rootsift),
    #         ' use_clahe_desc={},'.format(use_clahe_desc),
    #         ' use_upright={}'.format(use_upright))

    # Initialize feature extractor
    NUM_FIRST_DETECT = 100000000
    if use_upright_minus_minus:
        NUM_FIRST_DETECT = num_kp
    if use_lower_det_th:
        feature = cv2.xfeatures2d.SIFT_create(NUM_FIRST_DETECT, 
                                              contrastThreshold=-10000,
                                              edgeThreshold=-10000)
    else:
        feature = cv2.xfeatures2d.SIFT_create(NUM_FIRST_DETECT)

    # Load image, for detection
    if use_clahe_det:
        img_det, _ = load_image(img_path,
                                use_color_image=True,
                                crop_center=False)
        img_det = l_clahe(img_det)
    else:
        img_det, _ = load_image(img_path,
                                use_color_image=False,
                                crop_center=False)

    # Load image, for descriptor extraction
    if use_clahe_desc:
        img_desc, _ = load_image(img_path,
                                 use_color_image=True,
                                 crop_center=False)
        img_desc = l_clahe(img_desc)
    else:
        img_desc, _ = load_image(img_path,
                                 use_color_image=False,
                                 crop_center=False)

    # Get keypoints
    kp = feature.detect(img_det, None)

    # Compute descriptors
    if use_upright:
        unique_kp = []
        for i, x in enumerate(kp):
            if i > 0:
                if x.response == kp[i - 1].response:
                    continue
            x.angle = 0
            unique_kp.append(x)
        unique_kp, unique_desc = feature.compute(img_desc, unique_kp, None)
        top_resps = np.array([x.response for x in unique_kp])
        idxs = np.argsort(top_resps)[::-1]
        kp = np.array(unique_kp)[idxs[:min(len(unique_kp), num_kp)]]
        desc = unique_desc[idxs[:min(len(unique_kp), num_kp)]]
    else:
        kp, desc = feature.compute(img_desc, kp, None)

    # Use root-SIFT
    if use_rootsift:
        desc /= desc.sum(axis=1, keepdims=True) + 1e-8
        desc = np.sqrt(desc)

    # Convert opencv keypoints into our format
    kp, desc = convert_opencv_kp_desc(kp, desc, num_kp)

    result = {}
    result['kp'] = [p[0:2] for p in kp]
    result['scale'] = [p[2] for p in kp]
    result['angle'] = [p[3] for p in kp]
    result['score'] = [p[4] for p in kp]
    result['descs'] = desc
    return result
