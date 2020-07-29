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

import argparse
from schema import Schema, And, Use, Optional

from utils.queue_helper import get_cluster_name
from utils.io_helper import parse_json
from utils.path_helper import get_filter_match_file

# Global variables within the script
arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Arguments for system settings
arg = add_argument_group('System')
arg.add_argument('--conda_env',
                 type=str,
                 default='sfm',
                 help='Conda environment name')
arg.add_argument('--cc_account',
                 type=str,
                 default='def-kyi',
                 help='Account (CC only)')
arg.add_argument(
    '--cc_time',
    type=str,
    default='03:00',
    help='Expected wall time in HH:MM. Three hours by default to maximize '
    'turnaround on Compute Canada')
arg.add_argument('--slurm_jobs_path',
                 type=str,
                 default='./jobs',
                 help='Directory that contains slurm jobs')
arg.add_argument('--slurm_logs_path',
                 type=str,
                 default='./logs',
                 help='Directory that contains slurm outputs')
arg.add_argument('--run_mode',
                 type=str,
                 default='batch',
                 choices=['interactive', 'batch'],
                 help='interactive or batch job')
arg.add_argument('--parallel',
                 type=int,
                 default=1,
                 help='Number of jobs to run in parallel')
arg.add_argument('--num_opencv_threads',
                 type=int,
                 default=0,
                 help='Number of threads for OpenCV (0: default)')
arg.add_argument('--opencv_seed',
                 type=int,
                 default=42,
                 help='Random seed for OpenCV')

# Arguments for Dataset
arg = add_argument_group('Dataset')
arg.add_argument(
    '--num_max_set',
    type=int,
    default=100,
    help='Number of images in the maximum set (superset, defaults to 100)')

# Arguments for Benchmark
arg = add_argument_group('Benchmark')
arg.add_argument('--json_method',
                 type=str,
                 default='',
                 help='Configuration: refer to the documentation for details.')
arg.add_argument('--path_data',
                 type=str,
                 default='../data',
                 help='Directory holding benchmark data.')
arg.add_argument('--path_results',
                 type=str,
                 default='../benchmark-results',
                 help='Directory holding benchmark results.')
arg.add_argument('--path_visualization',
                 type=str,
                 default='../benchmark-visualization',
                 help='Directory holding visualization results.')
arg.add_argument('--path_pack',
                 type=str,
                 default='',
                 help='Directory holding benchmark results.')

arg.add_argument(
    '--vis_th',
    type=float,
    default=100,
    help='Threshold for determining valid pairs. Currently looks at '
    'visibility.txt. Note that as of now, we look at the number of '
    'shared keypoints.')
arg.add_argument(
    '--error_level',
    type=int,
    default=0,
    help='Level of error handling. 0 will warn and continue, 1 will '
    'throw error only on critical things, will throw error at any suspicion.')
arg.add_argument('--eval_stereo',
                 type=str2bool,
                 default=True,
                 help='Set to false to bypass the stereo task')
arg.add_argument('--eval_multiview',
                 type=str2bool,
                 default=True,
                 help='Set to false to bypass the multiview task')
arg.add_argument('--eval_relocalization',
                 type=str2bool,
                 default=True,
                 help='Set to false to bypass the relocalization task')

# Some configurations for colmap
arg.add_argument('--colmap_min_model_size',
                 type=int,
                 default=3,
                 help='Minimum size to be used for mapper')

# Some configurations for computing matching score
arg.add_argument(
    '--matching_score_epipolar_threshold',
    type=float,
    default=1e-4,
    help='Threshold for computing epipolar distance based matching score. '
    'In normalized camera coordinates, and with symmetric epipolar distance')
arg.add_argument(
    '--matching_score_reprojection_threshold',
    type=float,
    default=1e-3,
    help='Threshold for computing reprojection distance based matching score. '
    'In normalized camera coordinates, and with symmetric reprojection '
    'distance')
arg.add_argument('--matching_repeatability_radius_threshold',
                 type=float,
                 default=1e-2,
                 help='Threshold for radius in repeatability test'
                 'In normalized camera coordinates')
arg.add_argument('--matching_score_and_repeatability_px_threshold',
                 type=int,
                 nargs='*',
                 default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 help='Matching score and repeatability pixel threshold')

# Arguments for visualization
arg = add_argument_group('Visualization')
arg.add_argument('--run_viz',
                 type=str2bool,
                 default=True,
                 help=''
                 'Turn this on to generate visualization data')
arg.add_argument('--run_viz_debug',
                 type=str2bool,
                 default=False,
                 help=''
                 'Turn this on to generate visualization data for debugging')
arg.add_argument(
    '--skip_packing',
    type=str2bool,
    default=False,
    help=''
    'Turn this on to skip packing the results (useful when generating '
    'visualizations)')
arg.add_argument('--num_viz_stereo_pairs',
                 type=int,
                 default=10,
                 help='Number of stereo pairs to visualize (per scene)')
arg.add_argument(
    '--num_viz_stereo_pairs_debug',
    type=int,
    default=200,
    help='Number of stereo pairs to visualize (per scene, for debugging)')
arg.add_argument('--max_num_images_viz_multiview',
                 type=int,
                 default=10,
                 help='Clip the number of multiview visualizations per bag')
arg.add_argument('--num_viz_colmap_subsets_bagsize3',
                 type=int,
                 default=0,
                 help='Number of colmap subsets to visualize')
arg.add_argument('--num_viz_colmap_subsets_bagsize5',
                 type=int,
                 default=0,
                 help='Number of colmap subsets to visualize')
arg.add_argument('--num_viz_colmap_subsets_bagsize10',
                 type=int,
                 default=0,
                 help='Number of colmap subsets to visualize')
arg.add_argument('--num_viz_colmap_subsets_bagsize25',
                 type=int,
                 default=1,
                 help='Number of colmap subsets to visualize')
arg.add_argument('--viz_composite_vert',
                 type=str2bool,
                 default=True,
                 help='Compose stereo vertically (not horizontally)')
arg.add_argument('--subset',
                 type=str,
                 default=None,
                 choices=['val', 'test'],
                 help='Data subset: "val" or "test"')

# run settings
arg = add_argument_group('run settings')
arg.add_argument('--num_runs_val_stereo',
                 type=int,
                 default=1,
                 help='Number of validation runs (stereo)')
arg.add_argument('--num_runs_val_multiview',
                 type=int,
                 default=1,
                 help='Number of validation runs (multiview)')
arg.add_argument('--num_runs_val_relocalization',
                 type=int,
                 default=1,
                 help='Number of validation runs (relocalization)')
arg.add_argument('--num_runs_test_stereo',
                 type=int,
                 default=3,
                 help='Number of test runs (stereo)')
arg.add_argument('--num_runs_test_multiview',
                 type=int,
                 default=3,
                 help='Number of test runs (multiview)')
arg.add_argument('--num_runs_test_relocalization',
                 type=int,
                 default=1,
                 help='Number of test runs (relocalization)')

# Challenge settings
arg = add_argument_group('Challenge settings')
arg.add_argument('--is_challenge',
                 type=str2bool,
                 default=False,
                 help='Enable for challenge entries (more strict settings)')
arg.add_argument('--is_submission',
                 type=str2bool,
                 default=False,
                 help='Flag as user submission (not from organizers)')

# Arguments that are automatically set by benchmark
# These arguments should not be manually adjusted. Use at your own risk!
arg = add_argument_group('Auto')
arg.add_argument('--method_dict', type=parse_json, default='{}', help='')
arg.add_argument('--dataset', type=str, default='', help='')
arg.add_argument('--scene', type=str, default='', help='')
arg.add_argument('--run', type=str, default='', help='')
arg.add_argument('--bag_size', type=int, default=-1, help='')
arg.add_argument('--bag_id', type=int, default=-1, help='')
arg.add_argument('--task', type=str, default='', help='')
for dataset in ['phototourism']:
    for subset in ['val', 'test']:
        arg.add_argument('--scenes_{}_{}'.format(dataset, subset),
                         type=str,
                         default='',
                         help='')
        arg.add_argument('--splits_{}_{}'.format(dataset, subset),
                         type=str,
                         default='',
                         help='')
arg.add_argument('--json_deprecated_images',
                 type=str,
                 default="",
                 help="JSON file containing deprecated images")


def validate_method(method, is_challenge):
    '''Validate method configuration passed as a JSON file.'''

    # Define a dictionary schema
    # TODO would be nice to not copy-paste for multiple datasets
    schema = Schema({
        Optional('metadata'): {
            'publish_anonymously':
            bool,
            'authors':
            str,
            'contact_email':
            str,
            'method_name':
            str,
            'method_description':
            str,
            # 'descriptor_type': str,
            # 'descriptor_size': And(int, lambda v: v >= 1),
            Optional('link_to_website'):
            str,
            Optional('link_to_pdf'):
            str,
            Optional('under_review'):
            bool,
            Optional('under_review_override'):
            str,
        },
        'config_common': {
            'json_label': str,
            'keypoint': And(Use(str), lambda v: '_' not in v),
            'descriptor': And(Use(str), lambda v: '_' not in v),
            'num_keypoints': And(int, lambda v: v > 1),
        },
        Optional('config_phototourism_stereo'): {
            Optional('use_custom_matches'): bool,
            Optional('custom_matches_name'): str,
            Optional('matcher'): {
                'method':
                And(str, lambda v: v in ['nn']),
                'distance':
                And(str, lambda v: v.lower() in ['l1', 'l2', 'hamming']),
                'flann':
                bool,
                'num_nn':
                And(int, lambda v: v >= 1),
                'filtering': {
                    'type':
                    And(
                        str, lambda v: v.lower() in [
                            'none', 'snn_ratio_pairwise', 'snn_ratio_vs_last',
                            'fginn_ratio_pairwise'
                        ]),
                    Optional('threshold'):
                    And(Use(float), lambda v: 0 < v <= 1),
                    Optional('fginn_radius'):
                    And(Use(float), lambda v: 0 < v < 100.),
                },
                Optional('descriptor_distance_filter'): {
                    'threshold': And(Use(float), lambda v: v > 0),
                },
                'symmetric': {
                    'enabled':
                    And(bool),
                    Optional('reduce'):
                    And(str, lambda v: v.lower() in ['both', 'any']),
                },
            },
            Optional('outlier_filter'): {
                'method':
                And(Use(str), lambda v: v.lower() in ['none', 'cne-bp-nd']),
            },
            Optional('geom'): {
                'method':
                And(
                    str, lambda v: v.lower() in [
                        'cv2-ransac-f', 'cv2-ransac-e', 'cv2-lmeds-f',
                        'cv2-lmeds-e', 'cv2-7pt', 'cv2-8pt',
                        'cv2-patched-ransac-f', 'cmp-degensac-f',
                        'cmp-degensac-f-laf', 'cmp-gc-ransac-f',
                        'cmp-degensac-f-laf', 'cmp-gc-ransac-f',
                        'cmp-magsac-f', 'cmp-gc-ransac-e', 'skimage-ransac-f',
                        'intel-dfe-f'
                    ]),
                Optional('threshold'):
                And(Use(float), lambda v: v > 0),
                Optional('confidence'):
                And(Use(float), lambda v: v > 0),
                Optional('max_iter'):
                And(Use(int), lambda v: v > 0),
                Optional('postprocess'):
                And(Use(bool), lambda v: v is not None),
                Optional('error_type'):
                And(Use(str),
                    lambda v: v.lower() in ['sampson', 'symm_epipolar']),
                Optional('degeneracy_check'):
                bool,
            }
        },
        Optional('config_phototourism_multiview'): {
            Optional('use_custom_matches'): bool,
            Optional('custom_matches_name'): str,
            Optional('matcher'): {
                'method':
                And(str, lambda v: v in ['nn']),
                'distance':
                And(str, lambda v: v.lower() in ['l1', 'l2', 'hamming']),
                'flann':
                bool,
                'num_nn':
                And(int, lambda v: v >= 1),
                'filtering': {
                    'type':
                    And(
                        str, lambda v: v.lower() in [
                            'none', 'snn_ratio_pairwise', 'snn_ratio_vs_last',
                            'fginn_ratio_pairwise'
                        ]),
                    Optional('threshold'):
                    And(Use(float), lambda v: 0 < v <= 1),
                    Optional('fginn_radius'):
                    And(Use(float), lambda v: 0 < v < 100.),
                },
                Optional('descriptor_distance_filter'): {
                    'threshold': And(Use(float), lambda v: v > 0),
                },
                'symmetric': {
                    'enabled':
                    And(bool),
                    Optional('reduce'):
                    And(str, lambda v: v.lower() in ['both', 'any']),
                },
            },
            Optional('outlier_filter'): {
                'method':
                And(Use(str), lambda v: v.lower() in ['none', 'cne-bp-nd']),
            },
            Optional('colmap'): {},
        },
        Optional('config_phototourism_relocalization'): {},
    })

    schema.validate(method)

    # Check for metadata for challenge entries
    if is_challenge and not method['metadata']:
        raise ValueError('Must specify metadata')

    # Check what we are running
    do_pt_stereo = False if 'config_phototourism_stereo' not in method \
            else bool(method['config_phototourism_stereo'])
    do_pt_multiview = False if 'config_phototourism_multiview' not in method \
            else bool(method['config_phototourism_multiview'])
    do_pt_relocalization = False if 'config_phototourism_relocalization' not \
            in method else bool(method['config_phototourism_relocalization'])

    if do_pt_stereo:
        print('Running: Phototourism, stereo track')
    if do_pt_multiview:
        print('Running: Phototourism, multiview track')
    if do_pt_relocalization:
        print('Running: Phototourism, relocalization track')
    if not any([do_pt_stereo, do_pt_multiview, do_pt_relocalization]):
        raise ValueError('No tasks were specified')

    # Check for incorrect, missing, or redundant options
    for dataset in ['phototourism']:
        for task in ['stereo', 'multiview', 'relocalization']:
            cur_key = 'config_{}_{}'.format(dataset, task)
            if cur_key not in method:
                print('Key "{}" is empty -> skipping check'.format(cur_key))
                continue
            else:
                print('Validating key "{}"'.format(cur_key))

            # If dict is not empty, use_custom_matches should exist
            if method[cur_key] and (
                    'use_custom_matches' not in method[cur_key]):
                raise ValueError('Must specify use_custom_matches')

            # If using custom matches, we cannot specify further options
            if method[cur_key] and ('use_custom_matches' in method[cur_key]) \
                    and method[cur_key]['use_custom_matches']:

                if 'matcher' in method[cur_key] or 'outlier_filter' in method[
                        cur_key]:
                    raise ValueError(
                        'Cannot specify a matcher or outlier filter with '
                        'use_custom_matches=True')

            # Matcher and filter
            if 'matcher' in method[cur_key]:
                matcher = method[cur_key]['matcher']
                if matcher['symmetric']['enabled'] and 'reduce' not in \
                        matcher['symmetric']:
                    raise ValueError(
                        '[{}/{}] Must specify "reduce" if "symmetric" is enabled'
                        .format(dataset, task))

                # Check for redundant settings with custom matches
                if 'config_{}_stereo'.format(dataset) in method:
                    cur_config = method['config_{}_stereo'.format(dataset)]
                    if cur_config['use_custom_matches']:
                        if 'matcher' in cur_config or 'outlier_filter' in cur_config \
                                or 'geom' in cur_config:
                            raise ValueError(
                                '[{}/stereo] Found redundant settings with use_custom_matches=True'
                                .format(dataset))
                    else:
                        if 'matcher' not in cur_config or 'outlier_filter' not in \
                                cur_config or 'geom' not in cur_config:
                            raise ValueError(
                                '[{}/stereo] Missing required settings with use_custom_matches=False'
                                .format(dataset))

                    if cur_config['use_custom_matches']:
                        if 'matcher' in cur_config or 'outlier_filter' in cur_config \
                                or 'geom' in cur_config:
                            raise ValueError(
                                '[{}/stereo] Found redundant settings with use_custom_matches=True'
                            )

            # For stereo, check also geom
            if task == 'stereo' and \
                    'config_{}_stereo'.format(dataset) in method and \
                    'geom' in method['config_{}_stereo'.format(dataset)]:
                geom = method['config_{}_stereo'.format(dataset)]['geom']

                # Threshold for RANSAC
                if geom['method'].lower() in [
                        'cv2-ransac-f', 'cv2-ransac-e', 'cv2-patched-ransac-f',
                        'cmp-degensac-f', 'cmp-gc-ransac-f', 'cmp-gc-ransac-e',
                        'cmp-degensac-f-laf', 'cmp-magsac-f',
                        'skimage-ransac-f', 'intel-dfe-f'
                ]:
                    if 'threshold' not in geom:
                        raise ValueError(
                            '[{}] Must specify a threshold for this method'.
                            format(dataset))
                else:
                    if 'threshold' in geom:
                        raise ValueError(
                            '[{}] Cannot specify a threshold for this method'.
                            format(dataset))

                # Degeneracy check for RANSAC
                if geom['method'].lower() in [
                        'cmp-degensac-f', 'cmp-degensac-f-laf'
                ]:
                    if 'degeneracy_check' not in geom:
                        raise ValueError(
                            '[{}] Must indicate degeneracy check for this method'
                            .format(dataset))
                    if 'error_type' not in geom:
                        raise ValueError(
                            '[{}] Must indicate error type for this method'.
                            format(dataset))
                else:
                    if 'degeneracy_check' in geom:
                        raise ValueError(
                            '[{}] Cannot apply degeneracy check to this method'
                            .format(dataset))
                    if 'error_type' in geom:
                        raise ValueError(
                            '[{}] Cannot indicate error type for this method'.
                            format(dataset))

                # Confidence for RANSAC/LMEDS
                if geom['method'].lower() in [
                        'cv2-ransac-f',
                        'cv2-patched-ransac-f',
                        'cv2-ransac-e',
                        'cv2-lmeds-f',
                        'cv2-lmeds-e',
                        'cmp-degensac-f',
                        'cmp-degensac-f-laf',
                        'cmp-gc-ransac-f',
                        'cmp-gc-ransac-e',
                        'skimage-ransac-f',
                        'cmp-magsac-f',
                ]:
                    if 'confidence' not in geom:
                        raise ValueError(
                            '[{}] Must specify a confidence value for OpenCV or DEGENSAC'
                            .format(dataset))
                else:
                    if 'confidence' in geom:
                        raise ValueError(
                            '[{}] Cannot specify a confidence value for this method'
                            .format(dataset))

                # Maximum number of RANSAC iterations
                if geom['method'].lower() in [
                        'cv2-patched-ransac-f',
                        'cmp-degensac-f',
                        'cmp-degensac-f-laf',
                        'cmp-gc-ransac-f',
                        'cmp-gc-ransac-e',
                        'skimage-ransac-f',
                        'cmp-magsac-f',
                ]:
                    if 'max_iter' not in geom:
                        raise ValueError(
                            '[{}] Must indicate max_iter for this method'.
                            format(dataset))
                else:
                    if 'max_iter' in geom:
                        raise ValueError(
                            '[{}] Cannot indicate max_iter for this method'.
                            format(dataset))

                # DFE-specific
                if geom['method'].lower() in ['intel-dfe-f']:
                    if 'postprocess' not in geom:
                        raise ValueError(
                            '[{}] Must specify a postprocess flag for DFE'.
                            format(dataset))
                else:
                    if 'postprocess' in geom:
                        raise ValueError(
                            '[{}] Cannot specify a postprocess flag for this method'
                            .format(dataset))


def get_config():
    cfg, unparsed = parser.parse_known_args()

    if cfg.parallel != 1:
        raise RuntimeError('Parallel option untested!')

    # Overwrite the options by explicitly selecting val/test
    # Hacky but convenient
    if cfg.subset == 'val':
        cfg.path_pack = 'packed-val'
    elif cfg.subset == 'test':
        cfg.path_pack = 'packed-test'
    else:
        print(' -- WARNING: non-standard subset, will dump results into '
              'packed-debug')
        cfg.path_pack = 'packed-debug'

    # Overwrite deprecated images json path
    cfg.json_deprecated_images = 'json/deprecated_images.json'

    # Enforce challenge settings
    if cfg.is_challenge:
        if cfg.num_runs_test_stereo != 3 or \
                cfg.num_runs_test_multiview != 3 or \
                cfg.num_runs_test_relocalization != 3:
            raise ValueError('Violating pre-set runs on challenge mode!')

    # "Ignore" walltime on Google Cloud Compute and change some defaults
    if get_cluster_name() == 'gcp' and cfg.run_mode == 'batch':
        cfg.cc_time = '32:00:00'
        cfg.num_opencv_threads = 2
        cfg.cc_account = 'default'

    return cfg, unparsed


def print_usage():
    parser.print_usage()
