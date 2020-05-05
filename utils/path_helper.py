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
import numpy as np


def get_eval_path(mode, cfg):
    if mode == 'feature':
        return get_feature_path(cfg)
    elif mode == 'match':
        return get_match_path(cfg)
    elif mode == 'filter':
        return get_filter_path(cfg)
    elif mode == 'model':
        return get_geom_path(cfg)
    elif mode == 'stereo':
        return get_stereo_path(cfg)
    elif mode == 'multiview':
        return get_multiview_path(cfg)
    else:
        raise ValueError('Unknown job type')


def get_eval_file(mode, cfg, job_id=None):
    if job_id:
        return os.path.join(get_eval_path(mode, cfg),
                            '{}.{}'.format(job_id, mode))
    else:
        try:
            file_list = os.listdir(get_eval_path(mode, cfg))
            valid_file = [
                file for file in file_list if file.split('.')[-1] == mode
            ]
            if len(valid_file) == 0:
                return None
            elif len(valid_file) == 1:
                return os.path.join(get_eval_path(mode, cfg), valid_file[0])
            else:
                print('Should never be here')
                import IPython
                IPython.embed()
                return None
        except FileNotFoundError:
            os.makedirs(get_eval_path(mode, cfg))
            return None


def get_data_path(cfg):
    '''Returns where the per-dataset results folder is stored.

    TODO: This probably should be done in a neater way.
    '''

    # Get data directory for 'set_100'
    return os.path.join(cfg.path_data, cfg.dataset, cfg.scene,
                        'set_{}'.format(cfg.num_max_set))


def get_base_path(cfg):
    '''Returns where the per-dataset results folder is stored.'''

    return os.path.join(cfg.path_results, cfg.dataset, cfg.scene)


def get_feature_path(cfg):
    '''Returns where the keypoints and descriptor results folder is stored.

    Method names converted to lower-case.'''

    common = cfg.method_dict['config_common']
    return os.path.join(
        get_base_path(cfg),
        '{}_{}_{}'.format(common['keypoint'].lower(), common['num_keypoints'],
                          common['descriptor'].lower()))


def get_kp_file(cfg):
    '''Returns the path to the keypoint file.'''

    return os.path.join(get_feature_path(cfg), 'keypoints.h5')


def get_scale_file(cfg):
    '''Returns the path to the scale file.'''

    return os.path.join(get_feature_path(cfg), 'scales.h5')


def get_score_file(cfg):
    '''Returns the path to the score file.'''

    return os.path.join(get_feature_path(cfg), 'scores.h5')


def get_angle_file(cfg):
    '''Returns the path to the angle file.'''

    return os.path.join(get_feature_path(cfg), 'angles.h5')


def get_affine_file(cfg):
    '''Returns the path to the angle file.'''

    return os.path.join(get_feature_path(cfg), 'affine.h5')


def get_desc_file(cfg):
    '''Returns the path to the descriptor file.'''

    return os.path.join(get_feature_path(cfg), 'descriptors.h5')


def get_match_name(cfg):
    '''Return folder name for the matching model.

    Converted to lower-case to avoid conflicts.'''
    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)

    # simply return 'custom_matcher' if it is provided
    if cfg.method_dict[cur_key]['use_custom_matches']:
        return cfg.method_dict[cur_key]['custom_matches_name']

    # consturct matcher name
    matcher = cfg.method_dict[cur_key]['matcher']

    # Make a custom string for the matching folder
    label = []

    # Base name
    label += [matcher['method']]

    # flann/bf
    if matcher['flann']:
        label += ['flann']
    else:
        label += ['bf']

    # number of neighbours
    label += ['numnn-{}'.format(matcher['num_nn'])]

    # distance
    label += ['dist-{}'.format(matcher['distance'])]

    # 2-way matching
    if not matcher['symmetric']['enabled']:
        label += ['nosym']
    else:
        label += ['sym-{}'.format(matcher['symmetric']['reduce'])]

    # filtering
    if matcher['filtering']['type'] == 'none':
        label += ['nofilter']
    elif matcher['filtering']['type'].lower() in [
            'snn_ratio_pairwise', 'snn_ratio_vs_last'
    ]:
        # Threshold == 1 means no ratio test
        # It just makes writing the config files easier
        if matcher['filtering']['threshold'] == 1:
            label += ['nofilter']
        else:
            label += [
                'filter-{}-{}'.format(matcher['filtering']['type'],
                                      matcher['filtering']['threshold'])
            ]
    elif matcher['filtering']['type'].lower() == 'fginn_ratio_pairwise':
        label += [
            'filter-fginn-pairwise-{}-{}'.format(
                matcher['filtering']['threshold'],
                matcher['filtering']['fginn_radius'])
        ]
    else:
        raise ValueError('Unknown filtering type')

    # distance filtering
    if 'descriptor_distance_filter' in matcher:
        if 'threshold' in matcher['descriptor_distance_filter']:
            max_dist = matcher['descriptor_distance_filter']['threshold']
            label += ['maxdist-{:.03f}'.format(max_dist)]

    return '_'.join(label).lower()


def get_filter_path(cfg):
    '''Returns folder location for the outlier filter results.'''

    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)

    # Bypass this when using custom matches
    if cfg.method_dict[cur_key]['use_custom_matches']:
        return os.path.join(get_match_path(cfg), 'no_filter')

    # Otherwise, depends on the filter method
    outlier_filter = cfg.method_dict[cur_key]['outlier_filter']
    if outlier_filter['method'] in ['cne-bp-nd']:
        return os.path.join(get_match_path(cfg), outlier_filter['method'])
    elif outlier_filter['method'] == 'none':
        return os.path.join(get_match_path(cfg), 'no_filter')
    else:
        raise ValueError('Unknown outlier_filter type')


def get_match_path(cfg):
    '''Returns where the match results folder is stored.'''
    return os.path.join(get_feature_path(cfg), get_match_name(cfg))


def get_match_file(cfg):
    '''Returns the path to the match file.'''

    return os.path.join(get_match_path(cfg), 'matches.h5')


def get_match_cost_file(cfg):
    '''Returns the path to the match file.'''

    return os.path.join(get_match_path(cfg), 'matching_cost.h5')


def get_geom_name(cfg):
    '''Return folder name for the geometry model.

    Converted to lower-case to avoid conflicts.'''

    geom = cfg.method_dict['config_{}_{}'.format(cfg.dataset,
                                                 cfg.task)]['geom']
    method = geom['method'].lower()

    # This entry is a temporary fix
    if method == 'cv2-patched-ransac-f':
        label = '_'.join([
            method, 'th',
            str(geom['threshold']),
            'conf', str(geom['confidence']),
            'maxiter', str(geom['max_iter'])
        ])
    elif method in ['cv2-ransac-e', 'cv2-ransac-f']:
        label = '_'.join([
            method,
            'th',
            str(geom['threshold']),
            'conf',
            str(geom['confidence']),
        ])
    elif method in ['cmp-degensac-f', 'cmp-degensac-f-laf', 'cmp-gc-ransac-e']:
        label = '_'.join([
            method, 'th',
            str(geom['threshold']), 'conf',
            str(geom['confidence']), 'max_iter',
            str(geom['max_iter']), 'error',
            str(geom['error_type']), 'degencheck',
            str(geom['degeneracy_check'])
        ])
    elif method in ['cmp-gc-ransac-f', 'skimage-ransac-f', 'cmp-magsac-f']:
        label = '_'.join([
            method, 'th',
            str(geom['threshold']), 'conf',
            str(geom['confidence']), 'max_iter',
            str(geom['max_iter'])
        ])
    elif method in ['cv2-lmeds-e', 'cv2-lmeds-f']:
        label = '_'.join([method, 'conf', str(geom['confidence'])])
    elif method in ['intel-dfe-f']:
        label = '_'.join([
            method, 'th',
            str(geom['threshold']), 'postprocess',
            str(geom['postprocess'])
        ])
    elif method in ['cv2-7pt', 'cv2-8pt']:
        label = method
    else:
        raise ValueError('Unknown method for E/F estimation')

    return label.lower()


def get_geom_path(cfg):
    '''Returns where the match results folder is stored.'''

    geom_name = get_geom_name(cfg)
    return os.path.join(get_filter_path(cfg), 'stereo-fold-{}'.format(cfg.run),
                        geom_name)


def get_geom_file(cfg):
    '''Returns the path to the match file.'''

    return os.path.join(get_geom_path(cfg), 'essential.h5')


def get_geom_inl_file(cfg):
    '''Returns the path to the match file.'''
    return os.path.join(get_geom_path(cfg), 'essential_inliers.h5')


def get_geom_cost_file(cfg):
    '''Returns the path to the geom cost file.'''
    return os.path.join(get_geom_path(cfg), 'geom_cost.h5')


def get_cne_temp_path(cfg):
    return os.path.join(get_filter_path(cfg), 'temp_cne')

def get_filter_match_file_for_computing_model(cfg):
    filter_match_file = os.path.join(get_filter_path(cfg), 
        'matches_imported_stereo_{}.h5'.format(cfg.run))
    if os.path.isfile(filter_match_file):
        return filter_match_file
    else:
        return get_filter_match_file(cfg)

def get_filter_match_file(cfg):
    return os.path.join(get_filter_path(cfg), 'matches_inlier.h5')


def get_filter_cost_file(cfg):
    return os.path.join(get_filter_path(cfg), 'matches_inlier_cost.h5')


def get_cne_data_dump_path(cfg):
    return os.path.join(get_cne_temp_path(cfg), 'data_dump')


def get_stereo_path(cfg):
    '''Returns the path to where the stereo results are stored.'''

    return os.path.join(get_geom_path(cfg), 'set_{}'.format(cfg.num_max_set))


def get_stereo_pose_file(cfg, th=None):
    '''Returns the path to where the stereo pose file.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(get_stereo_path(cfg),
                        'stereo_pose_errors{}.h5'.format(label))


def get_repeatability_score_file(cfg, th=None):
    '''Returns the path to the repeatability file.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(get_stereo_path(cfg),
                        'repeatability_score_file{}.h5'.format(label))


def get_stereo_epipolar_pre_match_file(cfg, th=None):
    '''Returns the path to the match file.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(get_stereo_path(cfg),
                        'stereo_epipolar_pre_match_errors{}.h5'.format(label))


def get_stereo_epipolar_refined_match_file(cfg, th=None):
    '''Returns the path to the filtered match file.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(
        get_stereo_path(cfg),
        'stereo_epipolar_refined_match_errors{}.h5'.format(label))


def get_stereo_epipolar_final_match_file(cfg, th=None):
    '''Returns the path to the match file after RANSAC.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(
        get_stereo_path(cfg),
        'stereo_epipolar_final_match_errors{}.h5'.format(label))


def get_stereo_depth_projection_pre_match_file(cfg, th=None):
    '''Returns the path to the errors file for input matches.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(
        get_stereo_path(cfg),
        'stereo_projection_errors_pre_match{}.h5'.format(label))


def get_stereo_depth_projection_refined_match_file(cfg, th=None):
    '''Returns the path to the errors file for filtered matches.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(
        get_stereo_path(cfg),
        'stereo_projection_errors_refined_match{}.h5'.format(label))


def get_stereo_depth_projection_final_match_file(cfg, th=None):
    '''Returns the path to the errors file for final matches.'''

    label = '' if th is None else '-th-{:s}'.format(th)
    return os.path.join(
        get_stereo_path(cfg),
        'stereo_projection_errors_final_match{}.h5'.format(label))


def get_colmap_path(cfg):
    '''Returns the path to colmap results for individual bag.'''

    return os.path.join(get_multiview_path(cfg),
                        'bag_size_{}'.format(cfg.bag_size),
                        'bag_id_{:05d}'.format(cfg.bag_id))

def get_multiview_path(cfg):
    '''Returns the path to multiview folder.'''

    return os.path.join(get_filter_path(cfg),
                        'multiview-fold-{}'.format(cfg.run))

def get_colmap_mark_file(cfg):
    '''Returns the path to colmap flag.'''

    return os.path.join(get_colmap_path(cfg), 'colmap_has_run')


def get_colmap_pose_file(cfg):
    '''Returns the path to colmap pose files.'''

    return os.path.join(get_colmap_path(cfg), 'colmap_pose_errors.h5')


def get_colmap_output_path(cfg):
    '''Returns the path to colmap outputs.'''

    return os.path.join(get_colmap_path(cfg), 'colmap')


def get_colmap_temp_path(cfg):
    '''Returns the path to colmap working path.'''

    # TODO: Do we want to use slurm temp directory?
    return os.path.join(get_colmap_path(cfg), 'temp_colmap')


def parse_file_to_list(file_name, data_dir):
    '''
    Parses filenames from the given text file using the `data_dir`

    :param file_name: File with list of file names
    :param data_dir: Full path location appended to the filename

    :return: List of full paths to the file names
    '''

    fullpath_list = []
    with open(file_name, 'r') as f:
        while True:
            # Read a single line
            line = f.readline()
            # Check the `line` type
            if not isinstance(line, str):
                line = line.decode('utf-8')
            if not line:
                break
            # Strip `\n` at the end and append to the `fullpath_list`
            fullpath_list.append(os.path.join(data_dir, line.rstrip('\n')))
    return fullpath_list


def get_fullpath_list(data_dir, key):
    '''
    Returns the full-path lists to image info in `data_dir`

    :param data_dir: Path to the location of dataset
    :param key: Which item to retrieve from

    :return: Tuple containing fullpath lists for the key item
    '''
    # Read the list of images, homography and geometry
    list_file = os.path.join(data_dir, '{}.txt'.format(key))

    # Parse files to fullpath lists
    fullpath_list = parse_file_to_list(list_file, data_dir)

    return fullpath_list


def get_item_name_list(fullpath_list):
    '''Returns each item name in the full path list, excluding the extension'''

    return [os.path.splitext(os.path.basename(_s))[0] for _s in fullpath_list]


def get_stereo_viz_folder(cfg):
    '''Returns the path to the stereo visualizations folder.'''

    base = os.path.join(cfg.method_dict['config_common']['json_label'].lower(),
                        cfg.dataset, cfg.scene, 'stereo')

    return os.path.join(cfg.path_visualization, 'png', base), \
           os.path.join(cfg.path_visualization, 'jpg', base)


def get_colmap_viz_folder(cfg):
    '''Returns the path to the multiview visualizations folder.'''

    base = os.path.join(cfg.method_dict['config_common']['json_label'].lower(),
                        cfg.dataset, cfg.scene, 'multiview')

    return os.path.join(cfg.path_visualization, 'png', base), \
           os.path.join(cfg.path_visualization, 'jpg', base)


def get_stereo_viz_folder_debug(cfg):
    '''Returns the path to the stereo visualizations folder.'''

    base = os.path.join(cfg.method_dict['config_common']['json_label'].lower(),
                        cfg.dataset, cfg.scene, 'stereo')

    return os.path.join(cfg.path_visualization + '-debug', 'png', base), \
           os.path.join(cfg.path_visualization + '-debug', 'jpg', base)


def get_colmap_viz_folder_debug(cfg):
    '''Returns the path to the multiview visualizations folder.'''

    base = os.path.join(cfg.method_dict['config_common']['json_label'].lower(),
                        cfg.dataset, cfg.scene, 'multiview')

    return os.path.join(cfg.path_visualization, 'png', base), \
           os.path.join(cfg.path_visualization, 'jpg', base)


def get_pairs_per_threshold(data_dir):
    pairs = {}
    for th in np.arange(0, 1, 0.1):
        pairs['{:0.1f}'.format(th)] = np.load(
            '{}/new-vis-pairs/keys-th-{:0.1f}.npy'.format(data_dir, th))
    return pairs
