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
import os
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from time import time
from config import get_config, print_usage
from utils import pack_helper
from utils.io_helper import load_h5, load_json
from utils.path_helper import get_desc_file


def main(cfg):
    '''Main function. Takes config as input.
    '''

    # Back up config
    cfg_orig = deepcopy(cfg)
    method = cfg_orig.method_dict

    # Add config options to the dict
    master_dict = OrderedDict()
    master_dict['config'] = method

    # Add date
    master_dict['properties'] = OrderedDict()
    master_dict['properties'][
        'processing_date'] = pack_helper.get_current_date()
    print('Adding processing date: {}'.format(
        master_dict['properties']['processing_date']))

    # Add submission flag
    master_dict['properties'][
        'is_submission'] = cfg.is_submission
    print('Flagging as user submission: {}'.format(cfg.is_submission))

    # Add descriptor properties
    cfg_desc = deepcopy(cfg_orig)
    cfg_desc.dataset = 'phototourism'
    cfg_desc.scene = 'british_museum'
    try:
        descriptors_dict = load_h5(get_desc_file(cfg_desc))
        desc_type, desc_size, desc_nbytes = pack_helper.get_descriptor_properties(
            cfg_desc, descriptors_dict)
    except Exception:
        desc_type = 'none'
        desc_size = 0
        desc_nbytes = 0
    master_dict['properties']['descriptor_type'] = desc_type
    master_dict['properties']['descriptor_size'] = desc_size
    master_dict['properties']['descriptor_nbytes'] = desc_nbytes
    print('Adding descriptor properties: {} {} ({} bytes)'.format(
        master_dict['properties']['descriptor_size'],
        master_dict['properties']['descriptor_type'],
        master_dict['properties']['descriptor_nbytes']))

    # get deprecated image list
    deprecated_images_list = load_json(cfg.json_deprecated_images)

    # Read data and splits
    for dataset in ['phototourism']:

        setattr(cfg_orig, 'scenes_{}_{}'.format(dataset, cfg_orig.subset),
                './json/data/{}_{}.json'.format(dataset, cfg_orig.subset))
        setattr(cfg_orig, 'splits_{}_{}'.format(dataset, cfg_orig.subset),
                './json/bag_size/{}_{}.json'.format(dataset, cfg_orig.subset))

        # Create empty dictionary
        master_dict[dataset] = OrderedDict()
        res_dict = OrderedDict()
        master_dict[dataset]['results'] = res_dict

        # Save number of runs
        master_dict[dataset]['num_runs_stereo'] = getattr(
            cfg_orig, 'num_runs_{}_stereo'.format(cfg_orig.subset))
        master_dict[dataset]['num_runs_multiview'] = getattr(
            cfg_orig, 'num_runs_{}_multiview'.format(cfg_orig.subset))

        # Load data config
        scene_list = load_json(
            getattr(cfg_orig, 'scenes_{}_{}'.format(dataset, cfg_orig.subset)))
        bag_size_json = load_json(
            getattr(cfg_orig, 'splits_{}_{}'.format(dataset, cfg_orig.subset)))
        bag_size_list = [b['bag_size'] for b in bag_size_json]
        bag_size_num = [b['num_in_bag'] for b in bag_size_json]
        bag_size_str = ['{}bag'.format(b) for b in bag_size_list]

        # Create empty dicts
        for scene in ['allseq'] + scene_list:
            res_dict[scene] = OrderedDict()
            for task in ['stereo', 'multiview', 'relocalization']:
                res_dict[scene][task] = OrderedDict()
                res_dict[scene][task]['run_avg'] = OrderedDict()
                if task == 'multiview':
                    for bag in bag_size_str + ['bag_avg']:
                        res_dict[scene]['multiview']['run_avg'][
                            bag] = OrderedDict()

        # Stereo -- multiple runs
        t = time()
        cur_key = 'config_{}_stereo'.format(dataset)
        if cfg_orig.eval_stereo and cur_key in method and method[cur_key]:
            num_runs = getattr(cfg_orig,
                               'num_runs_{}_stereo'.format(cfg_orig.subset))
            cfg = deepcopy(cfg_orig)
            cfg.dataset = dataset
            cfg.task = 'stereo'
            for scene in scene_list:

                # get deprecated images
                if scene in deprecated_images_list.keys():
                    deprecated_images = deprecated_images_list[scene]
                else:
                    deprecated_images = []

                cfg.scene = scene

                res_dict[scene]['stereo']['run_avg'] = OrderedDict()
                for run in range(num_runs):
                    res_dict[scene]['stereo']['run_{}'.format(
                        run)] = OrderedDict()

                # Create list of things to gather
                metric_list = []
                metric_list += ['avg_num_keypoints']
                # metric_list += ['matching_scores_epipolar']
                metric_list += ['num_inliers']
                metric_list += ['matching_scores_depth_projection']
                metric_list += ['repeatability']
                metric_list += ['qt_auc']
                metric_list += ['timings']

                for run in range(num_runs):
                    # Compute and pack results
                    cfg.run = run
                    cur_dict = res_dict[scene]['stereo']['run_{}'.format(run)]
                    for metric in metric_list:
                        t_cur = time()
                        getattr(pack_helper, 'compute_' + metric)(cur_dict,
                                                                  deprecated_images, cfg)
                        print(
                            ' -- Packing "{}"/"{}"/stereo, run: {}/{}, metric: {} [{:.02f} s]'
                            .format(dataset, scene, run + 1, num_runs, metric,
                                    time() - t_cur))

            # Compute average across runs, for stereo
            t_cur = time()
            pack_helper.average_stereo_over_runs(cfg, res_dict, num_runs)
            print(
                ' -- Packing "{}"/stereo: averaging over {} run(s) [{:.02f} s]'
                .format(dataset, num_runs,
                        time() - t_cur))

            # Compute average across scenes, for stereo
            t_cur = time()
            pack_helper.average_stereo_over_scenes(cfg, res_dict, num_runs)
            print(
                ' -- Packing "{}"/stereo: averaging over {} scene(s) [{:.02f} s]'
                .format(dataset, len(scene_list),
                        time() - t_cur))

            print(' -- Finished packing stereo in {:.01f} sec.'.format(time() -
                                                                       t))
        else:
            print('Skipping "{}/stereo"'.format(dataset))

        # Multiview -- multiple runs
        t = time()
        cur_key = 'config_{}_multiview'.format(dataset)
        if cfg_orig.eval_multiview and cur_key in method and method[cur_key]:
            num_runs = getattr(cfg, 'num_runs_{}_multiview'.format(cfg.subset))
            cfg = deepcopy(cfg_orig)
            cfg.dataset = dataset
            cfg.task = 'multiview'
            for scene in scene_list:
                cfg.scene = scene
                
                # get deprecated images
                if scene in deprecated_images_list.keys():
                    deprecated_images = deprecated_images_list[scene]
                else:
                    deprecated_images = []

                for run in ['run_avg'
                            ] + ['run_{}'.format(f) for f in range(num_runs)]:
                    res_dict[scene]['multiview'][run] = OrderedDict()
                    for bags_label in ['bag_avg'] + bag_size_str:
                        res_dict[scene]['multiview'][run][
                            bags_label] = OrderedDict()

                # Create list of things to gather
                metric_list = []
                metric_list += ['avg_num_keypoints']
                metric_list += ['num_input_matches']
                metric_list += ['qt_auc_colmap']
                metric_list += ['ATE']
                metric_list += ['colmap_stats']

                for run in range(num_runs):
                    for bag_size in bag_size_list:
                        # Compute and pack results
                        cfg.run = run
                        cfg.bag_size = bag_size
                        cur_dict = res_dict[scene]['multiview']
                        for metric in metric_list:
                            t_cur = time()
                            getattr(pack_helper, 'compute_' + metric)(
                                cur_dict['run_{}'.format(run)]['{}bag'.format(
                                    bag_size)], deprecated_images, cfg)
                            print(
                                ' -- Packing "{}"/"{}"/multiview, run {}/{}, "{}", metric: {} [{:.02f} s]'
                                .format(dataset, scene, run + 1, num_runs,
                                        '{}bag'.format(bag_size), metric,
                                        time() - t_cur))

                        # Compute average across bags
                        for metric in cur_dict['run_{}'.format(run)]['25bag']:
                            pack_helper.average_multiview_over_bags(
                                cfg, cur_dict['run_{}'.format(run)],
                                bag_size_list)

            # Compute average across runs, for multiview
            t_cur = time()
            pack_helper.average_multiview_over_runs(cfg, res_dict, num_runs,
                                                    bag_size_str + ['bag_avg'])
            print(
                ' -- Packing "{}"/multiview: averaging over {} run(s) [{:.02f} s]'
                .format(dataset, num_runs,
                        time() - t_cur))

            # Compute average across scenes, for multiview
            t_cur = time()
            pack_helper.average_multiview_over_scenes(
                cfg, res_dict, num_runs, ['bag_avg'] + bag_size_str)
            print(
                ' -- Packing "{}"/multiview: averaging over {} scene(s) [{:.02f} s]'
                .format(dataset, len(scene_list),
                        time() - t_cur))

            print(' -- Finished packing multiview in {:.01f} sec.'.format(
                time() - t))

            # Relocalization -- multiple runs
            # TODO
        else:
            print('Skipping "{}/multiview"'.format(dataset))

    # Dump packed result
    print(' -- Saving to: "{}"'.format(
        cfg.method_dict['config_common']['json_label']))
    if not os.path.exists(cfg.path_pack):
        os.makedirs(cfg.path_pack)
    json_dump_file = os.path.join(
        cfg.path_pack,
        '{}.json'.format(cfg.method_dict['config_common']['json_label']))

    with open(json_dump_file, 'w') as outfile:
        json.dump(master_dict, outfile, indent=2)


if __name__ == '__main__':
    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(cfg)
