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
from copy import deepcopy
import numpy as np
from shutil import copyfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

from config import get_config, print_usage
from utils.io_helper import load_h5, load_json
from utils.load_helper import load_image, load_h5_valid_image
from utils.path_helper import (get_kp_file, get_colmap_output_path,
                               get_colmap_viz_folder)
from utils.colmap_helper import (get_best_colmap_index,
                                 get_colmap_image_path_list, valid_bag)
from third_party.colmap.scripts.python.read_write_model import (
    read_images_binary, read_points3d_binary)


def main(cfg):
    '''Visualization of colmap points.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    '''

    bag_size_json = load_json(
        getattr(cfg, 'splits_{}_{}'.format(cfg.dataset, cfg.subset)))
    bag_size_list = [b['bag_size'] for b in bag_size_json]
    bag_size_num = [b['num_in_bag'] for b in bag_size_json]

    # Load deprecated images list
    deprecated_images_list = load_json(cfg.json_deprecated_images)
    if cfg.scene in deprecated_images_list.keys():
        deprecated_images = deprecated_images_list[cfg.scene]
    else:
        deprecated_images = []

    # # Do not re-run if files already exist -- off for now
    # skip = True
    # for _bag_size in bag_size_list:
    #     cfg_bag = deepcopy(cfg)
    #     cfg_bag.bag_size = _bag_size
    #     viz_folder_hq, viz_folder_lq = get_colmap_viz_folder(cfg_bag)
    #     for _bag_id in range(
    #             getattr(cfg_bag,
    #                     'num_viz_colmap_subsets_bagsize{}'.format(_bag_size))):
    #         if any([
    #                 not os.path.exists(
    #                     os.path.join(
    #                         viz_folder_lq,
    #                         'colmap-bagsize{:d}-bag{:02d}-image{:02d}.jpg'.
    #                         format(_bag_size, _bag_id, i)))
    #                 for i in range(_bag_size)
    #         ]):
    #             skip = False
    #             break
    #         if not os.path.exists(
    #                 os.path.join(
    #                     viz_folder_lq,
    #                     'colmap-bagsize{:d}-bag{:02d}.pcd'.format(
    #                         _bag_size, _bag_id))):
    #             skip = False
    #             break
    # if skip:
    #     print(' -- already exists, skipping colmap visualization')
    #     return

    print(' -- Visualizations, multiview: "{}/{}"'.format(
        cfg.dataset, cfg.scene))
    t_start = time()

    # Create results folder if it does not exist
    for _bag_size in bag_size_list:
        cfg_bag = deepcopy(cfg)
        cfg_bag.bag_size = _bag_size
        viz_folder_hq, viz_folder_lq = get_colmap_viz_folder(cfg_bag)
        if not os.path.exists(viz_folder_hq):
            os.makedirs(viz_folder_hq)
        if not os.path.exists(viz_folder_lq):
            os.makedirs(viz_folder_lq)

    # Load keypoints
    keypoints_dict = load_h5(get_kp_file(cfg))

    # Loop over bag sizes
    for _bag_num, _bag_size in zip(bag_size_num, bag_size_list):
        cfg_bag = deepcopy(cfg)
        cfg_bag.bag_size = _bag_size
        num_bags = getattr(
            cfg_bag, 'num_viz_colmap_subsets_bagsize{}'.format(_bag_size))

        # select valid bag
        valid_bag_ids = []
        bag_id = 0
        while len(valid_bag_ids) < num_bags:
            cfg_bag.bag_id = bag_id
            if valid_bag(cfg_bag, deprecated_images):
                valid_bag_ids.append(bag_id)
            bag_id = bag_id + 1
            if bag_id == _bag_num:
                raise RuntimeError('Ran out of bags to check out')

        for _bag_id in valid_bag_ids:
            print(
                ' -- Visualizations, multiview: "{}/{}", bag_size={}, bag {}/{}'
                .format(cfg.dataset, cfg.scene, _bag_size, _bag_id + 1,
                        num_bags))

            # Retrieve list of images
            cfg_bag.bag_id = _bag_id
            images_in_bag = get_colmap_image_path_list(cfg_bag)

            # Retrieve reconstruction
            colmap_output_path = get_colmap_output_path(cfg_bag)
            # is_colmap_valid = os.path.exists(
            #     os.path.join(colmap_output_path, '0'))
            best_index = get_best_colmap_index(cfg_bag)
            if best_index != -1:
                colmap_images = read_images_binary(
                    os.path.join(colmap_output_path, str(best_index),
                                 'images.bin'))
            for i, image_path in enumerate(images_in_bag):
                # Limit to 10 or so, even for bag size 25
                if i >= cfg.max_num_images_viz_multiview:
                    break

                # Load image and keypoints
                im, _ = load_image(image_path,
                                   use_color_image=True,
                                   crop_center=False,
                                   force_rgb=True)
                used = None
                key = os.path.splitext(os.path.basename(image_path))[0]
                if best_index != -1:
                    for j in colmap_images:
                        if key in colmap_images[j].name:
                            # plot all keypoints
                            used = colmap_images[j].point3D_ids != -1
                            break
                if used is None:
                    used = [False] * keypoints_dict[key].shape[0]
                used = np.array(used)

                fig = plt.figure(figsize=(20, 20))
                plt.imshow(im)
                plt.plot(keypoints_dict[key][~used, 0],
                         keypoints_dict[key][~used, 1],
                         'r.',
                         markersize=12)
                plt.plot(keypoints_dict[key][used, 0],
                         keypoints_dict[key][used, 1],
                         'b.',
                         markersize=12)
                plt.tight_layout()
                plt.axis('off')

                # TODO Ideally we would save to pdf
                # but it does not work on 16.04, so we do png instead
                # https://bugs.launchpad.net/ubuntu/+source/imagemagick/+bug/1796563
                viz_folder_hq, viz_folder_lq = get_colmap_viz_folder(cfg_bag)
                viz_file_hq = os.path.join(
                    viz_folder_hq,
                    'bagsize{:d}-bag{:02d}-image{:02d}.png'.format(
                        _bag_size, _bag_id, i))
                viz_file_lq = os.path.join(
                    viz_folder_lq,
                    'bagsize{:d}-bag{:02d}-image{:02d}.jpg'.format(
                        _bag_size, _bag_id, i))
                plt.savefig(viz_file_hq, bbox_inches='tight')

                # Convert with imagemagick
                os.system('convert -quality 75 -resize \"640>\" {} {}'.format(
                    viz_file_hq, viz_file_lq))

                plt.close()

            if best_index != -1:
                colmap_points = read_points3d_binary(
                    os.path.join(colmap_output_path, str(best_index),
                                 'points3D.bin'))
                points3d = []
                for k in colmap_points:
                    points3d.append([
                        colmap_points[k].xyz[0], colmap_points[k].xyz[1],
                        colmap_points[k].xyz[2]
                    ])
                points3d = np.array(points3d)
                points3d -= np.median(points3d, axis=0)[None, ...]
                points3d /= np.abs(points3d).max() + 1e-6
                pcd = os.path.join(
                    get_colmap_viz_folder(cfg_bag)[0],
                    'colmap-bagsize{:d}-bag{:02d}.pcd'.format(
                        _bag_size, _bag_id))
                with open(pcd, 'w') as f:
                    f.write('# .PCD v.7 - Point Cloud Data file format\n')
                    f.write('VERSION .7\n')
                    f.write('FIELDS x y z\n')
                    f.write('SIZE 4 4 4\n')
                    f.write('TYPE F F F\n')
                    f.write('COUNT 1 1 1\n')
                    f.write('WIDTH {}\n'.format(len(colmap_points)))
                    f.write('HEIGHT 1\n')
                    f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
                    f.write('POINTS {}\n'.format(len(colmap_points)))
                    f.write('DATA ascii\n')
                    for p in points3d:
                        f.write('{:.05f} {:.05f} {:.05f}\n'.format(
                            p[0], p[1], p[2]))
                copyfile(
                    os.path.join(
                        get_colmap_viz_folder(cfg_bag)[0],
                        'colmap-bagsize{:d}-bag{:02d}.pcd'.format(
                            _bag_size, _bag_id)),
                    os.path.join(
                        get_colmap_viz_folder(cfg_bag)[1],
                        'colmap-bagsize{:d}-bag{:02d}.pcd'.format(
                            _bag_size, _bag_id)))

    print('done [{:.02f} s.]'.format(time() - t_start))


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(cfg)
