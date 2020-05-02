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

from copy import deepcopy
import os

from config import get_config, print_usage, validate_method
from utils.colmap_helper import is_colmap_complete
from utils.io_helper import load_json
from utils.queue_helper import (create_and_queue_jobs, create_sh_cmd,
                                estimate_runtime, is_job_complete,
                                create_job_key)


def create_eval_jobs(dep_list, mode, cfg, job_dict):
    # Check if job is complete
    if is_job_complete(mode, cfg):
        print(' -- File {} already exists'.format(mode))
        return []

    # Check if other program is doing the same job
    job_key = create_job_key(mode, cfg)
    if job_key in job_dict:
        print(' -- {} is already running on {}'.format(mode,
                                                       job_dict[job_key]))
        return job_dict[job_key].split('-')
    else:
        # Update dependency
        dep_str = None
        if len(dep_list) > 0:
            dep_str = ','.join(dep_list)
        # Check if matches are computed -- queue (dependent on previous
        # job)
        print(' -- Computing {}'.format(mode))
        cmd_list = [create_sh_cmd('compute_{}.py'.format(mode), cfg)]
        job = create_and_queue_jobs(cmd_list, cfg, dep_str)
        job_dict[job_key] = job
        return [job]


def eval_viz_stereo(dep_list, cfg, debug=False):
    # Do this one for one run
    if cfg.run > 0:
        return

    # Update dependency
    dep_str = None
    if len(dep_list) > 0:
        dep_str = ','.join(dep_list)

    # The checks on existing files run inside, as there are many of them
    if debug:
        print(' -- Generating stereo visualizations (debug)')
        cmd_list = [create_sh_cmd('viz_stereo_debug.py', cfg)]
    else:
        print(' -- Generating stereo visualizations')
        cmd_list = [create_sh_cmd('viz_stereo.py', cfg)]
    create_and_queue_jobs(cmd_list, cfg, dep_str)


def eval_viz_colmap(dep_list, cfg):
    # Do this one for one run
    if cfg.run > 0:
        return

    # Update dependency
    dep_str = None
    if len(dep_list) > 0:
        dep_str = ','.join(dep_list)

    # The checks on existing files run inside, as there are many of them
    print(' -- Generating multi-view visualizations')
    cmd_list = [create_sh_cmd('viz_colmap.py', cfg)]
    create_and_queue_jobs(cmd_list, cfg, dep_str)


def eval_packing(dep_list, cfg):
    # Update dependency
    dep_str = None
    if len(dep_list) > 0:
        dep_str = ','.join(dep_list)

    print(' -- Packing results')
    cmd_list = [create_sh_cmd('pack_res.py', cfg)]
    create_and_queue_jobs(cmd_list, cfg, dep_str)


def eval_multiview(dep_list, cfg, bag_size_list, bag_size_num, job_dict):
    colmap_jobs = []
    job_key = create_job_key('multiview', cfg)
    # Update dependency
    dep_str = None
    if len(dep_list) > 0:
        dep_str = ','.join(dep_list)
    # COLMAP evaluation
    #
    # TODO; For colmap, should we queue twice?
    cfg_bag = deepcopy(cfg)
    cmd_list = []
    cfg_list = []
    print(' -- The multiview task  will work on these bags {}'.format([
        '{} (x{})'.format(b, n) for b, n in zip(bag_size_list, bag_size_num)
    ]))
    for _bag_size, _num_in_bag in zip(bag_size_list, bag_size_num):
        for _bag_id in range(_num_in_bag):
            cfg_bag.bag_size = _bag_size
            cfg_bag.bag_id = _bag_id

            # Check if colmap evaluation is complete -- queue
            if not is_colmap_complete(cfg_bag):
                # Check if other program is doing the same job
                if job_key in job_dict:
                    print(' -- {} is already running on {}'.format('multiview',
                            job_dict[job_key]))
                    return job_dict[job_key].split('-')

                cmd_list += [create_sh_cmd('eval_colmap.py', cfg_bag)]
                cfg_list += [deepcopy(cfg_bag)]
            else:
                print(' -- Multiview: bag size {} bag id {} results'
                      ' already exists'.format(_bag_size, _bag_id))
            # Check cfg_list to retrieve the estimated runtime. Queue
            # cmd_list and reset both lists if we are expected to have
            # less than 30 min of wall time after this job.
            t_split = [float(t) for t in cfg.cc_time.split(':')]
            if estimate_runtime(cfg_list) >= t_split[0] + \
                    t_split[1] / 60 - 0.5:
                colmap_jobs += [create_and_queue_jobs(cmd_list, cfg, dep_str)]
                cmd_list = []
                cfg_list = []
    # Queue any leftover jobs for this bag
    if len(cmd_list) > 0:
        colmap_jobs += [create_and_queue_jobs(cmd_list, cfg, dep_str)]
    # save colmap jobs list under its job key
    if len(colmap_jobs)!=0:
        job_dict[job_key] = '-'.join(colmap_jobs)
    return colmap_jobs


def main(cfg):
    ''' Main routine for the benchmark '''

    # Read data and splits
    for dataset in ['phototourism']:
        for subset in ['val', 'test']:
            setattr(cfg, 'scenes_{}_{}'.format(dataset, subset),
                    './json/data/{}_{}.json'.format(dataset, subset))
            setattr(cfg, 'splits_{}_{}'.format(dataset, subset),
                    './json/bag_size/{}_{}.json'.format(dataset, subset))

    # Read the list of methods and datasets
    method_list = load_json(cfg.json_method)
    for i, method in enumerate(method_list):
        print('Validating method {}/{}: "{}"'.format(
            i + 1, len(method_list), method['config_common']['json_label']))
        validate_method(method, is_challenge=cfg.is_challenge)

    # Back up original config
    cfg_orig = deepcopy(cfg)
    job_dict = {}

    # Loop over methods, datasets/scenes, and tasks
    for method in method_list:
        # accumulate packing dependencies over datasets and runs
        all_stereo_jobs = []
        all_multiview_jobs = []
        all_relocalization_jobs = []

        for dataset in ['phototourism']:
            # Load data config
            scene_list = load_json(
                getattr(cfg_orig,
                        'scenes_{}_{}'.format(dataset, cfg_orig.subset)))
            bag_size_json = load_json(
                getattr(cfg_orig,
                        'splits_{}_{}'.format(dataset, cfg_orig.subset)))
            bag_size_list = [b['bag_size'] for b in bag_size_json]
            bag_size_num = [b['num_in_bag'] for b in bag_size_json]

            for scene in scene_list:
                print('Working on {}: {}/{}'.format(
                    method['config_common']['json_label'], dataset, scene))

                # For each task
                for task in ['stereo', 'multiview', 'relocalization']:
                    # Skip if the key does not exist or it is empty
                    cur_key = 'config_{}_{}'.format(dataset, task)
                    if cur_key not in method or not method[cur_key]:
                        print(
                            'Empty config for "{}", skipping!'.format(cur_key))
                        continue

                    # Append method to config
                    cfg = deepcopy(cfg_orig)
                    cfg.method_dict = deepcopy(method)
                    cfg.dataset = dataset
                    cfg.task = task
                    cfg.scene = scene

                    # Features
                    feature_jobs = create_eval_jobs([], 'feature', cfg,
                                                    job_dict)

                    # Matches
                    match_jobs = create_eval_jobs(feature_jobs, 'match', cfg,
                                                  job_dict)

                    # Filter
                    match_inlier_jobs = create_eval_jobs(
                        match_jobs, 'filter', cfg, job_dict)

                    # Empty dependencies
                    stereo_jobs = []
                    multiview_jobs = []
                    relocalization_jobs = []

                    num_runs = getattr(
                        cfg, 'num_runs_{}_{}'.format(cfg.subset, task))
                    for run in range(num_runs):
                        cfg.run = run

                        # Pose estimation and stereo evaluation
                        if task == 'stereo' and cfg.eval_stereo:
                            geom_model_jobs = create_eval_jobs(
                                match_inlier_jobs, 'model', cfg, job_dict)
                            stereo_jobs += create_eval_jobs(
                                geom_model_jobs, 'stereo', cfg, job_dict)
                            all_stereo_jobs += stereo_jobs

                        # Visualization for stereo
                        if task == 'stereo' and cfg.run_viz:
                            eval_viz_stereo(stereo_jobs, cfg)

                        # Debugging for stereo
                        if task == 'stereo' and cfg.run_viz_debug:
                            eval_viz_stereo(stereo_jobs, cfg, debug=True)

                        # Multiview
                        if task == 'multiview' and cfg.eval_multiview:
                            multiview_jobs += eval_multiview(
                                match_inlier_jobs, cfg, bag_size_list,
                                bag_size_num, job_dict)
                            all_multiview_jobs += multiview_jobs

                        # Visualization for colmap
                        if task == 'multiview' and cfg.run_viz:
                            eval_viz_colmap(multiview_jobs, cfg)

                        if task == 'relocalization' and cfg.eval_relocalization:
                            raise NotImplementedError(
                                'TODO relocalization task')

        # Packing -- can be skipped with --skip_packing=True
        # For instance, when only generating visualizations
        if not cfg.skip_packing:
            cfg = deepcopy(cfg_orig)
            cfg.method_dict = deepcopy(method)
            eval_packing(
                all_stereo_jobs + all_multiview_jobs + all_relocalization_jobs,
                cfg)


if __name__ == '__main__':

    cfg, unparsed = get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(cfg)
