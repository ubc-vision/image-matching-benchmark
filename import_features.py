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
import h5py
import os
import numpy as np
from shutil import copy
from glob import glob
import json
from utils.io_helper import load_json
from utils.pack_helper import get_descriptor_properties


def get_kp_category(num_kp):
    '''Determine category by number of keypoints.'''

    breakpoints = [2048, 8000]
    for b in breakpoints:
        if num_kp <= b:
            return b

    raise RuntimeError('Found more keypoints than allowed')


def get_desc_category(desc_nbytes):
    '''Determine category by descriptor size.'''

    # 128 bytes -> 32 float32
    # 512 bytes -> 128 float32
    # 2048 bytes -> 512 float32
    breakpoints = [128, 512, 2048]
    for b in breakpoints:
        if desc_nbytes <= b:
            return b

    raise RuntimeError('Descriptors are larger than allowed')


def get_descriptor_nbytes(cfg, data_list):
    '''Return descriptor size in bytes.'''

    descriptor
    for _data in data_list:
        print('--- On "{}"...'.format(_data))
        with h5py.File(os.path.join(cfg.path_features, _data, 'keypoints.h5'),
                       'r') as f_kp:
            for k in f_kp:
                size_kp_file.append(f_kp[k].shape[0])


def validate_label(label):
    if '_' in label:
        print('WARNING: Replacing underscore with hyphen in method name')
    return label.replace('_', '-').lower()


def import_features(cfg, data_list):
    '''
    Import features with the third (2020) submission format (separate files for
    scores, orientations, and scales). Keypoint category is set by looking at
    the maximum value over all files, instead of the mean.
    '''

    # Retrieve stats
    print('Retrieving number of keypoints...')
    size_kp_file = []
    for _data in data_list:
        print('--- On "{}"...'.format(_data))
        with h5py.File(os.path.join(cfg.path_features, _data, 'keypoints.h5'),
                       'r') as f_kp:
            for k in f_kp:
                size_kp_file.append(f_kp[k].shape[0])

    if len(size_kp_file) == 0:
        raise RuntimeError('No keypoints?')

    print('------ Found {} keypoint files'.format(len(size_kp_file)))
    print('------ Min: {}, max: {}, mean: {}'.format(np.min(size_kp_file),
                                                     np.max(size_kp_file),
                                                     np.mean(size_kp_file)))

    # If no category is selected, determine it automatically
    if cfg.num_keypoints == -1:
        numkp = get_kp_category(np.max(size_kp_file))
        print('Setting number of keypoints category to: {}'.format(numkp))
    # Otherwise, hand-pick it
    else:
        numkp = cfg.num_keypoints
        print('Pre-selected number of keypoints: {}'.format(numkp))

    # only check descriptor size if it is provided
    if os.path.isfile(os.path.join(cfg.path_features, data_list[0], 'descriptors.h5')):
        # Open a descriptors file to get their size
        print('Retrieving descriptor_size...')
        for _data in data_list:
            print('--- On "{}"...'.format(_data))
            with h5py.File(
                    os.path.join(cfg.path_features, _data, 'descriptors.h5'),
                    'r') as f_desc:
                desc_type, desc_size, desc_nbytes = get_descriptor_properties(
                    cfg, f_desc)
                break
            break
        print('Descriptor type: {} {} ({} bytes)'.format(desc_size, desc_type,
                                                         desc_nbytes))
        nbytes_category = get_desc_category(desc_nbytes)
        print('Falling under challenge category: {} bytes'.format(nbytes_category))
    else:
        print('Descriptor file is not given')
    # Import
    print('Importing features...')
    for _data in data_list:
        print('--- On "{}"...'.format(_data))

        fn_kp = os.path.join(cfg.path_features, _data, 'keypoints.h5')
        fn_desc = os.path.join(cfg.path_features, _data, 'descriptors.h5')
        fn_score = os.path.join(cfg.path_features, _data, 'scores.h5')
        fn_scale = os.path.join(cfg.path_features, _data, 'scales.h5')
        fn_ori = os.path.join(cfg.path_features, _data, 'orientations.h5')
        fn_match = os.path.join(cfg.path_features, _data, 'matches.h5')
        fn_multiview_match = os.path.join(cfg.path_features, _data, 'matches_multiview.h5')
        fn_stereo_match_list = [os.path.join(cfg.path_features, _data,'matches_stereo_{}.h5').
            format(idx) for idx in range(3)]

        # create keypoints folder
        tgt_cur = os.path.join(
            cfg.path_results, _data,
            '_'.join([cfg.kp_name, str(numkp), cfg.desc_name]))
        if not os.path.isdir(tgt_cur):
            os.makedirs(tgt_cur)

        # Both keypoints and descriptors files are provided
        if os.path.isfile(fn_kp) and os.path.isfile(fn_desc) and not \
           (os.path.isfile(fn_match) or (
           (os.path.isfile(fn_multiview_match) and os.path.isfile(fn_stereo_match_list[0])))):
            # We cannot downsample the keypoints without scores
            if numkp < max(size_kp_file) and not os.path.isfile(fn_score):
                raise RuntimeError('------ No scores, and subsampling is required!'
                                   '(wanted: {}, found: {})'.format(
                                       numkp, max(size_kp_file)))

            # Import keypoints
            print('------ Importing keypoints and descriptors')

            # If there is no need to subsample, we can just copy the files
            if numkp >= max(size_kp_file):
                copy(fn_kp, tgt_cur)
                copy(fn_desc, tgt_cur)
                if os.path.isfile(fn_score):
                    copy(fn_score, tgt_cur)
                if os.path.isfile(fn_scale):
                    copy(fn_scale, tgt_cur)
                if os.path.isfile(fn_ori):
                    copy(fn_ori, tgt_cur)
            # Otherwise, crop each file separately
            else:
                subsampled_indices = {}
                with h5py.File(fn_score, 'r') as h5_r, \
                        h5py.File(os.path.join(tgt_cur, 'scores.h5'), 'w') as h5_w:
                    for k in h5_r:
                        sorted_indices = np.argsort(h5_r[k])[::-1]
                        subsampled_indices[k] = sorted_indices[:min(
                            h5_r[k].size, numkp)]
                        crop = h5_r[k].value[subsampled_indices[k]]
                        h5_w[k] = crop
                with h5py.File(fn_kp, 'r') as h5_r, \
                        h5py.File(
                                os.path.join(tgt_cur, 'keypoints.h5'),
                                'w') as h5_w:
                    for k in h5_r:
                        crop = h5_r[k].value[subsampled_indices[k], :]
                        h5_w[k] = crop
                with h5py.File(fn_desc, 'r') as h5_r, \
                        h5py.File(
                                os.path.join(
                                    tgt_cur, 'descriptors.h5'), 'w') as h5_w:
                    for k in h5_r:
                        crop = h5_r[k].value[subsampled_indices[k], :]
                        h5_w[k] = crop
                if os.path.isfile(fn_scale):
                    with h5py.File(fn_scale, 'r') as h5_r, \
                            h5py.File(
                                    os.path.join(tgt_cur, 'scales.h5'),
                                    'w') as h5_w:
                        for k in h5_r:
                            crop = h5_r[k].value[subsampled_indices[k]]
                            h5_w[k] = crop
                if os.path.isfile(fn_ori):
                    with h5py.File(fn_ori, 'r') as h5_r, \
                            h5py.File(
                                    os.path.join(tgt_cur, 'orientations.h5'),
                                    'w') as h5_w:
                        for k in h5_r:
                            crop = h5_r[k].value[subsampled_indices[k]]
                            h5_w[k] = crop
        elif os.path.isfile(fn_kp) and \
             (os.path.isfile(fn_match) or \
             (os.path.isfile(fn_multiview_match) and os.path.isfile(fn_stereo_match_list[0]))):

            if os.path.isfile(fn_desc):
                print('------ Matches file is provided')
            print('------ Importing matches')
            if not cfg.match_name:
                raise RuntimeError('Must define match_name')

            # For match only submission, no downsampling is performed.
            if numkp < max(size_kp_file):
                raise RuntimeError('------ number of keypoints exceeds maximum allowed limit'
                                   '(wanted: {}, found: {})'.format(
                                       numkp, max(size_kp_file)))

            # copy keypoints file to raw results folder
            copy(fn_kp, tgt_cur)
            if os.path.isfile(fn_desc):
                print(fn_desc)
                print(tgt_cur)
                copy(fn_desc, tgt_cur)
            if os.path.isfile(fn_score):
                copy(fn_score, tgt_cur)
            if os.path.isfile(fn_scale):
                copy(fn_scale, tgt_cur)
            if os.path.isfile(fn_ori):
                copy(fn_ori, tgt_cur)

            # create match folder with match method name
            match_folder_path = os.path.join(tgt_cur,cfg.match_name)
            if not os.path.isdir(match_folder_path):
                os.makedirs(match_folder_path)
            # copy match file to raw results folder

            if os.path.isfile(fn_multiview_match) and os.path.isfile(fn_stereo_match_list[0]):
                print('------ Multiview match file and Stereo match file are provided seperately')
                fn_match = fn_multiview_match
            else:
                print('------ Only one match file is provided for both stereo and multiview tasks')

            copy(fn_match,os.path.join(match_folder_path,'matches.h5'))
            # make dummy cost file
            with h5py.File(os.path.join(match_folder_path,'matching_cost.h5'),'w') as h5_w:
                h5_w.create_dataset('cost', data=0.0)

            # create post filter folder with 'no filter'
            filter_folder_path = os.path.join(match_folder_path,'no_filter')
            if not os.path.isdir(filter_folder_path):
                os.makedirs(filter_folder_path)
            # copy match file to post filter folder
            copy(fn_match,os.path.join(filter_folder_path,'matches_inlier.h5'))
            # make dummy cost file
            with h5py.File(os.path.join(filter_folder_path,'matches_inlier_cost.h5'),'w') as h5_w:
                h5_w.create_dataset('cost', data=0.0)

            # check if three stereo matches are provided
            if all([os.path.isfile(fn_stereo_match_list[idx]) for idx in range(3)]):
                print('------ Three stereo match files are provided')
            # if only one stereo match is provided, copy it three times
            elif os.path.isfile(fn_stereo_match_list[0]):
                print('------ One stereo match files is provided, copy it three times')
                fn_stereo_match_list = [fn_stereo_match_list[0]]*3
            # if only one match is provided for both stereo and multiview, copy it three times
            else:
                fn_stereo_match_list = [fn_match]*3

            for idx, fn_stereo_match in enumerate(fn_stereo_match_list):
                copy(fn_stereo_match,
                    os.path.join(filter_folder_path,'matches_imported_stereo_{}.h5'.format(idx)))
        else:
            raise RuntimeError('Neither descriptors nor matches are provided!')


        # Preserved for debugging custom matches

        # if os.path.isfile(fn_match):
        #     print('------ Importing matches')

        #     if not cfg.match_name:
        #         raise RuntimeError('Must define match_name')

        #     tgt_cur = os.path.join(
        #         cfg.path_results,
        #         _data, '_'.join([cfg.kp_name,
        #                          str(numkp), cfg.desc_name]), cfg.match_name)

        #     if not os.path.isdir(tgt_cur):
        #         os.makedirs(tgt_cur)
        #     if cfg.matches_key_reverse:
        #         print('------ Reversing key')
        #         with h5py.File(fn_match, 'w') as h5_w:
        #             with h5py.File(
        #                     os.path.join(cfg.path_features, _data,
        #                                  'matches.h5'), 'r') as h5_r:
        #                 keys = [key for key in h5_r.keys()]
        #                 for key in keys:
        #                     split_key = key.split('-')
        #                     new_key = split_key[1] + '-' + split_key[0]
        #                     h5_w[new_key] = h5_r[key].value
        #                     h5_w[new_key][1, :] = h5_r[key].value[0, :]
        #                     h5_w[new_key][0, :] = h5_r[key].value[1, :]
        #     else:
        #         copy(fn_match, tgt_cur)
        # else:
        #     print('------ No custom match file!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--matches_key_reverse',
        action='store_true',
        default=False,
        help='reverse the image name position in match keys')
    parser.add_argument(
        '--kp_name',
        type=str,
        default='',
        help='Name of the method used to extract keypoints, lower case only')
    parser.add_argument(
        '--desc_name',
        type=str,
        default='',
        help='Name of the method used to extract descriptors, lower case only')
    parser.add_argument(
        '--match_name',
        type=str,
        default='',
        help='Name of the method used to match features, if any, '
        'lower case only')
    parser.add_argument(
        '--num_keypoints',
        type=int,
        default=-1,
        help='Number of keypoints (-1 to use all)')
    parser.add_argument(
        '--path_features',
        type=str,
        help='Path to the features to import')
    parser.add_argument(
        '--path_results',
        type=str,
        default='../benchmark-results/phototourism/',
        help='Directory holding benchmark results.')
    parser.add_argument(
        '--subset',
        type=str,
        default='both',
        help='Subset to import: "val", "test", "both" (default), "spc-fix"')

    cfg, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        parser.print_usage()
        exit(-1)

    if not cfg.kp_name:
        raise RuntimeError('Must define kp_name')
    if not cfg.desc_name:
        raise RuntimeError('Must define desc_name')
    if cfg.match_name and cfg.num_keypoints != -1:
        raise RuntimeError('Can not crop keypoints list with a custom matcher')

    cfg.kp_name = validate_label(cfg.kp_name)
    cfg.desc_name = validate_label(cfg.desc_name)
    cfg.match_name = validate_label(cfg.match_name)

    seqs = []
    if cfg.subset == 'spc-fix':
        seqs += ['st_pauls_cathedral']
    elif cfg.subset in ['val', 'test', 'both']:
        if cfg.subset in ['val', 'both']:
            with open(os.path.join('json', 'data',
                                   'phototourism_val.json')) as f:
                seqs += json.load(f)
        if cfg.subset in ['test', 'both']:
            with open(os.path.join('json', 'data',
                                   'phototourism_test.json')) as f:
                seqs += json.load(f)
    else:
        raise ValueError('Invalid subset')
    print('Processing the following scenes: {}'.format(seqs))

    print('Importing, kp:"{}", desc="{}", matcher="{}", num_keypoints="{}" '.
          format(cfg.kp_name, cfg.desc_name,
                 cfg.match_name if cfg.match_name else 'N/A',
                 cfg.num_keypoints if cfg.num_keypoints != -1 else 'N/A'))

    import_features(cfg, seqs)
