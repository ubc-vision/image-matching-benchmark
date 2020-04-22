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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from time import time

from config import get_config, print_usage
from utils.io_helper import load_h5, build_composite_image, load_json
from utils.path_helper import (get_data_path, get_kp_file, get_match_file,
                               get_stereo_epipolar_final_match_file,
                               get_stereo_viz_folder, get_geom_file,
                               get_geom_inl_file)
from utils.load_helper import load_depth, load_calib, load_h5_valid_image
from utils.stereo_helper import (np_skew_symmetric, get_projected_kp,
                                 normalize_keypoints, unnormalize_keypoints,
                                 get_truesym)


def main(cfg):
    '''Visualization of stereo keypoints and matches.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    '''

    # Files should not be named to prevent (easy) abuse
    # Instead we use 0, ..., cfg.num_viz_stereo_pairs
    viz_folder_hq, viz_folder_lq = get_stereo_viz_folder(cfg)

    # # Do not re-run if files already exist -- off for now
    # if os.path.exists(viz_folder_lq):
    #     if all([
    #             os.path.exists(
    #                 os.path.join(viz_folder_lq, 'stereo-{}.jpg'.format(i)))
    #             for i in range(cfg.num_viz_stereo_pairs)
    #     ]):
    #         print(' -- already exists, skipping stereo visualization')
    #         return

    print(' -- Visualizations, stereo: "{}/{}"'.format(cfg.dataset, cfg.scene))
    t_start = time()

    # Load deprecated images list
    deprecated_images_list = load_json(cfg.json_deprecated_images)
    if cfg.scene in deprecated_images_list.keys():
        deprecated_images = deprecated_images_list[cfg.scene]
    else:
        deprecated_images = []

    # Load keypoints, matches and errors
    keypoints_dict = load_h5_valid_image(get_kp_file(cfg), deprecated_images)
    matches_dict = load_h5_valid_image(get_match_file(cfg), deprecated_images)

    # Hacky: We need to recompute the errors, loading only for the keys
    errors_dict = load_h5_valid_image(
        get_stereo_epipolar_final_match_file(cfg, th='0.1'), deprecated_images)

    # Get data directory
    data_dir = get_data_path(cfg)

    # Create results folder if it does not exist
    if not os.path.exists(viz_folder_hq):
        os.makedirs(viz_folder_hq)
    if not os.path.exists(viz_folder_lq):
        os.makedirs(viz_folder_lq)

    # Sort alphabetically and pick different images
    sorted_keys = sorted(errors_dict)
    picked = []
    pairs = []
    for pair in sorted_keys:
        fn1, fn2 = pair.split('-')
        if fn1 not in picked and fn2 not in picked:
            picked += [fn1, fn2]
            pairs += [pair]
        if len(pairs) == cfg.num_viz_stereo_pairs:
            break

    # Load all depth maps
    depth = {}
    for pair in pairs:
        files = pair.split('-')
        for f in files:
            if f not in depth:
                depth[f] = load_depth(
                    os.path.join(data_dir, 'depth_maps', '{}.h5'.format(f)))

    # Generate and save the images
    for i, pair in enumerate(pairs):
        # load metadata
        fn1, fn2 = pair.split('-')
        calib_dict = load_calib([
            os.path.join(data_dir, 'calibration',
                         'calibration_{}.h5'.format(fn1)),
            os.path.join(data_dir, 'calibration',
                         'calibration_{}.h5'.format(fn2))
        ])
        calc1 = calib_dict[fn1]
        calc2 = calib_dict[fn2]
        matches = matches_dict[pair]
        ransac_inl_dict = load_h5_valid_image(get_geom_inl_file(cfg),
                                              deprecated_images)
        inl = ransac_inl_dict[pair]

        # Get depth for keypoints
        kp1 = keypoints_dict[fn1]
        kp2 = keypoints_dict[fn2]
        kp1_int = np.round(kp1).astype(int)
        kp2_int = np.round(kp2).astype(int)

        kp1_int[:, 1] = np.clip(kp1_int[:, 1], 0, depth[fn1].shape[0] - 1)
        kp1_int[:, 0] = np.clip(kp1_int[:, 0], 0, depth[fn1].shape[1] - 1)
        kp2_int[:, 1] = np.clip(kp2_int[:, 1], 0, depth[fn2].shape[0] - 1)
        kp2_int[:, 0] = np.clip(kp2_int[:, 0], 0, depth[fn2].shape[1] - 1)
        d1 = np.expand_dims(depth[fn1][kp1_int[:, 1], kp1_int[:, 0]], axis=-1)
        d2 = np.expand_dims(depth[fn2][kp2_int[:, 1], kp2_int[:, 0]], axis=-1)

        # Get {R, t} from calibration information
        R_1, t_1 = calc1['R'], calc1['T'].reshape((3, 1))
        R_2, t_2 = calc2['R'], calc2['T'].reshape((3, 1))

        # Compute dR, dt
        dR = np.dot(R_2, R_1.T)
        dT = t_2 - np.dot(dR, t_1)

        # Normalize keypoints
        kp1n = normalize_keypoints(kp1, calc1['K'])
        kp2n = normalize_keypoints(kp2, calc2['K'])

        # Project with depth
        kp1n_p, kp2n_p = get_projected_kp(kp1n, kp2n, d1, d2, dR, dT)
        kp1_p = unnormalize_keypoints(kp1n_p, calc2['K'])
        kp2_p = unnormalize_keypoints(kp2n_p, calc1['K'])

        # Re-index keypoints from matches
        kp1_inl = kp1[inl[0]]
        kp2_inl = kp2[inl[1]]
        kp1_p_inl = kp1_p[inl[0]]
        kp2_p_inl = kp2_p[inl[1]]
        kp1n_inl = kp1n[inl[0]]
        kp2n_inl = kp2n[inl[1]]
        kp1n_p_inl = kp1n_p[inl[0]]
        kp2n_p_inl = kp2n_p[inl[1]]
        d1_inl = d1[inl[0]]
        d2_inl = d2[inl[1]]

        # Filter out keypoints with invalid depth
        nonzero_index = np.nonzero(np.squeeze(d1_inl * d2_inl))
        zero_index = np.where(np.squeeze(d1_inl * d2_inl) == 0)[0]
        kp1_inl_nonzero = kp1_inl[nonzero_index]
        kp2_inl_nonzero = kp2_inl[nonzero_index]
        kp1_p_inl_nonzero = kp1_p_inl[nonzero_index]
        kp2_p_inl_nonzero = kp2_p_inl[nonzero_index]
        kp1n_inl_nonzero = kp1n_inl[nonzero_index]
        kp2n_inl_nonzero = kp2n_inl[nonzero_index]
        kp1n_p_inl_nonzero = kp1n_p_inl[nonzero_index]
        kp2n_p_inl_nonzero = kp2n_p_inl[nonzero_index]

        # Compute symmetric distance using the depth image
        true_d = get_truesym(kp1_inl_nonzero, kp2_inl_nonzero,
                             kp1_p_inl_nonzero, kp2_p_inl_nonzero)

        # canvas
        im, v_offset, h_offset = build_composite_image(
            os.path.join(data_dir, 'images', fn1 + '.jpg'),
            os.path.join(data_dir, 'images', fn2 + '.jpg'),
            margin=5,
            axis=0 if cfg.viz_composite_vert else 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        linewidth = 2

        # Plot matches on points without depth
        for idx in range(len(zero_index)):
            plt.plot(
                (kp1_inl[idx, 0] + h_offset[0], kp2_inl[idx, 0] + h_offset[1]),
                (kp1_inl[idx, 1] + v_offset[0], kp2_inl[idx, 1] + v_offset[1]),
                color='b',
                linewidth=linewidth)

        # Plot matches on points with depth
        max_dist = 5
        cmap = matplotlib.cm.get_cmap('summer')
        order = list(range(len(true_d)))
        random.shuffle(order)
        for idx in order:
            if true_d[idx] <= max_dist:
                min_val = 0
                max_val = 255 - min_val
                col = cmap(
                    int(max_val * (1 - (max_dist - true_d[idx]) / max_dist) +
                        min_val))
                # col = cmap(255 * (max_dist - true_d[idx]) / max_dist)
            else:
                col = 'r'
            plt.plot((kp1_inl_nonzero[idx, 0] + h_offset[0],
                      kp2_inl_nonzero[idx, 0] + h_offset[1]),
                     (kp1_inl_nonzero[idx, 1] + v_offset[0],
                      kp2_inl_nonzero[idx, 1] + v_offset[1]),
                     color=col,
                     linewidth=linewidth)

        plt.tight_layout()
        plt.axis('off')
        viz_file_hq = os.path.join(viz_folder_hq, '{:05d}.png'.format(i))
        viz_file_lq = os.path.join(viz_folder_lq, '{:05d}.jpg'.format(i))
        plt.savefig(viz_file_hq, bbox_inches='tight')

        # Convert with imagemagick
        os.system('convert -quality 75 -resize \"500>\" {} {}'.format(
            viz_file_hq, viz_file_lq))

        plt.close()

    print('done [{:.02f} s.]'.format(time() - t_start))


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(cfg)
