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
import hashlib
import re
import itertools
import json
from utils.io_helper import load_json
from utils.pack_helper import get_descriptor_properties

def hash_folder(folder_path):
    hash = hashlib.md5()
    hash = get_hash_list(folder_path,hash)
    return hash.hexdigest()[:16]

def get_hash_list(folder_path,hash):

    files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]
    dirs = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,f))]

    for _file in sorted(files):
        if _file.endswith('.h5') or os.path.basename(_file)=='config.json':
            with open(_file, 'rb') as fp:
                hash.update(fp.read())
    for _dir in sorted(dirs):
        hash = get_hash_list(_dir,hash)

    return hash

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
    breakpoints = [128, 512, 2048, 4096, 8192]
    for b in breakpoints:
        if desc_nbytes <= b:
            return b

    raise RuntimeError('Descriptors are larger than allowed')


def get_descriptor_nbytes(cfg, seq_list):
    '''Return descriptor size in bytes.'''

    descriptor
    for _seq in seq_list:
        print('--- On "{}"...'.format(_seq))
        with h5py.File(os.path.join(cfg.path_features, _seq, 'keypoints.h5'),
                       'r') as f_kp:
            for k in f_kp:
                size_kp_file.append(f_kp[k].shape[0])


def validate_label(label):
    if '_' in label:
        print('WARNING: Replacing underscore with hyphen in method name')
    return label.replace('_', '-').lower()

def reformat_json(path_src_json, path_tar_json):
    config = load_json(path_src_json)
    with open(path_tar_json, 'w') as f:
        json.dump(config, f, indent=2)

def add_hash_prfix(path_src_json,path_tar_json,hash_str):
    with open(path_src_json,'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'json_label' in line:
                if not hash_str in line:
                    lines[idx] = re.sub('"(?P<w1>.*?)"(?P<w2>[^"]+)"(?P<w3>.*?)"', '"\g<w1>"\g<w2>"{}-\g<w3>"'.format(hash_str), line)

    with open(path_tar_json, 'w') as f:
        f.writelines(lines)

def import_features(cfg):
    '''
    Import features with the third (2020) submission format (separate files for
    scores, orientations, and scales). Keypoint category is set by looking at
    the maximum value over all files, instead of the mean.
    '''

    # Get subset sequences
    seqs_dict = {}
    for _dataset in cfg.datasets:
        seqs = []
        if cfg.subset in ['val', 'both']:
            with open(os.path.join('json', 'data',
                                   f'{_dataset}_val.json')) as f:
                seqs += json.load(f)
        if cfg.subset in ['test', 'both']:
            with open(os.path.join('json', 'data',
                                   f'{_dataset}_test.json')) as f:
                seqs += json.load(f)
        seqs_dict[_dataset] = seqs

    # Retrieve stats
    print('Retrieving number of keypoints...')
    size_kp_file = []
    for _dataset in cfg.datasets:
        for _seq in seqs_dict[_dataset]:
            print('--- On "{}:{}"...'.format(_dataset, _seq))
            with h5py.File(os.path.join(cfg.path_features, _dataset, _seq, 'keypoints.h5'),
                           'r') as f_kp:
                for k in f_kp:
                    size_kp_file.append(f_kp[k].shape[0])

    if len(size_kp_file) == 0:
        raise RuntimeError('No keypoints?')

    print('------ Found {} keypoint files'.format(len(size_kp_file)))
    print('------ Min: {}, max: {}, mean: {}'.format(np.min(size_kp_file),
                                                     np.max(size_kp_file),
                                                     np.mean(size_kp_file)))

    # Set number of kp to a large number for pairwise matching
    if cfg.pairwise_matching:
        numkp = 1000000
        print('Pairwise matching mode, no limit on kp')
    # If no category is selected, determine it automatically
    elif cfg.num_keypoints == -1:
        numkp = get_kp_category(np.max(size_kp_file))
        print('Setting number of keypoints category to: {}'.format(numkp))
    # Otherwise, hand-pick it
    else:
        numkp = cfg.num_keypoints
        print('Pre-selected number of keypoints: {}'.format(numkp))

    # only check descriptor size if it is provided
    if os.path.isfile(os.path.join(cfg.path_features, cfg.datasets[0], seqs_dict[cfg.datasets[0]][0], 'descriptors.h5')):
        # Open a descriptors file to get their size
        print('Retrieving descriptor_size...')
        for _dataset in cfg.datasets:
            for _seq in seqs_dict[_dataset]:
                print('--- On "{} {}"...'.format(_dataset, _seq))
                with h5py.File(
                        os.path.join(cfg.path_features, _dataset, _seq, 'descriptors.h5'),
                        'r') as f_desc:
                    desc_type, desc_size, desc_nbytes = get_descriptor_properties(
                        cfg, f_desc)
                    break
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
    for _dataset in cfg.datasets:
        for _seq in seqs_dict[_dataset]:
            print('--- On "{}: {}"...'.format(_dataset,_seq))

            fn_kp = os.path.join(cfg.path_features, _dataset, _seq, 'keypoints.h5')
            fn_desc = os.path.join(cfg.path_features, _dataset, _seq, 'descriptors.h5')
            fn_score = os.path.join(cfg.path_features, _dataset, _seq, 'scores.h5')
            fn_scale = os.path.join(cfg.path_features, _dataset, _seq, 'scales.h5')
            fn_ori = os.path.join(cfg.path_features, _dataset, _seq, 'orientations.h5')
            fn_match = os.path.join(cfg.path_features, _dataset, _seq, 'matches.h5')
            fn_multiview_match = os.path.join(cfg.path_features, _dataset, _seq, 'matches_multiview.h5')
            fn_stereo_match = os.path.join(cfg.path_features, _dataset, _seq,'matches_stereo.h5')
            fn_stereo_match_list = [os.path.join(cfg.path_features, _dataset, _seq,'matches_stereo_{}.h5').
                format(idx) for idx in range(3)]

            # create keypoints folder
            if cfg.pairwise_matching:
                tgt_cur = os.path.join(
                    cfg.path_results, _dataset, _seq,
                    '_'.join([cfg.kp_name, str(-1), cfg.desc_name]))
            else:
                tgt_cur = os.path.join(
                    cfg.path_results, _dataset, _seq,
                    '_'.join([cfg.kp_name, str(numkp), cfg.desc_name]))
            if not os.path.isdir(tgt_cur):
                os.makedirs(tgt_cur)

            # Both keypoints and descriptors files are provided
            if os.path.isfile(fn_kp) and os.path.isfile(fn_desc) and not \
               (os.path.isfile(fn_match) or 
               (os.path.isfile(fn_multiview_match) and (os.path.isfile(fn_stereo_match_list[0]) or os.path.isfile(fn_stereo_match)))):
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
                                sorted_indices = np.argsort(h5_r[k].value.reshape(-1))[::-1]
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
                 (os.path.isfile(fn_multiview_match) and
                 (os.path.isfile(fn_stereo_match) or os.path.isfile(fn_stereo_match_list[0])))):
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
                if isinstance(cfg.match_name,dict):
                    # hardcode task names!
                    multiview_match_folder_path = os.path.join(tgt_cur,cfg.match_name[_dataset+'_multiview'])
                    stereo_match_folder_path = os.path.join(tgt_cur,cfg.match_name[_dataset+'_stereo'])
                else:
                    multiview_match_folder_path = os.path.join(tgt_cur,cfg.match_name)
                    stereo_match_folder_path = os.path.join(tgt_cur,cfg.match_name)
                if not os.path.isdir(multiview_match_folder_path):
                    os.makedirs(multiview_match_folder_path)
                if not os.path.isdir(stereo_match_folder_path):
                    os.makedirs(stereo_match_folder_path)

                # copy match file to raw results folder
                if os.path.isfile(fn_multiview_match) and \
                   (os.path.isfile(fn_stereo_match) or os.path.isfile(fn_stereo_match_list[0])):
                    print('------ Multiview match file and Stereo match file are provided seperately')
                    if not os.path.isfile(fn_stereo_match_list[1]):
                        print('------ Only one stereo match file is provided, copy it three times')
                        if os.path.isfile(fn_stereo_match_list[0]):
                            fn_stereo_match_list = [fn_stereo_match_list[0]]*3
                        else:
                            fn_stereo_match_list = [fn_stereo_match]*3
                    else:
                        print('------ Three stereo match files are provided')
                else:
                    print('------ Only one match file is provided for both stereo and multiview tasks')
                    fn_multiview_match = fn_match
                    fn_stereo_match_list = [fn_match]*3
                copy(fn_multiview_match,os.path.join(multiview_match_folder_path,'matches.h5'))
                copy(fn_stereo_match_list[0],os.path.join(stereo_match_folder_path,'matches.h5'))

                # make dummy cost file
                with h5py.File(os.path.join(multiview_match_folder_path,'matching_cost.h5'),'w') as h5_w:
                    h5_w.create_dataset('cost', data=0.0)
                with h5py.File(os.path.join(stereo_match_folder_path,'matching_cost.h5'),'w') as h5_w:
                    h5_w.create_dataset('cost', data=0.0)

                # create post filter folder with 'no filter'
                stereo_filter_folder_path = os.path.join(stereo_match_folder_path,'no_filter')
                if not os.path.isdir(stereo_filter_folder_path):
                    os.makedirs(stereo_filter_folder_path)
                multiview_filter_folder_path = os.path.join(multiview_match_folder_path,'no_filter')
                if not os.path.isdir(multiview_filter_folder_path):
                    os.makedirs(multiview_filter_folder_path)

                # copy match file to post filter folder
                copy(fn_multiview_match,os.path.join(multiview_filter_folder_path,'matches_inlier.h5'))
                copy(fn_stereo_match_list[0],os.path.join(stereo_filter_folder_path,'matches_inlier.h5'))

                # make dummy cost file
                with h5py.File(os.path.join(stereo_filter_folder_path,'matches_inlier_cost.h5'),'w') as h5_w:
                    h5_w.create_dataset('cost', data=0.0)
                with h5py.File(os.path.join(multiview_filter_folder_path,'matches_inlier_cost.h5'),'w') as h5_w:
                    h5_w.create_dataset('cost', data=0.0)

                for idx, fn_stereo_match in enumerate(fn_stereo_match_list):
                    copy(fn_stereo_match,
                        os.path.join(stereo_filter_folder_path,'matches_imported_stereo_{}.h5'.format(idx)))
            else:
                raise RuntimeError('Neither descriptors nor matches are provided!')


        # Preserved for debugging custom matches

        # if os.path.isfile(fn_match):
        #     print('------ Importing matches')

        #     if not cfg.match_name:
        #         raise RuntimeError('Must define match_name')

        #     tgt_cur = os.path.join(
        #         cfg.path_results,
        #         _seq, '_'.join([cfg.kp_name,
        #                          str(numkp), cfg.desc_name]), cfg.match_name)

        #     if not os.path.isdir(tgt_cur):
        #         os.makedirs(tgt_cur)
        #     if cfg.matches_key_reverse:
        #         print('------ Reversing key')
        #         with h5py.File(fn_match, 'w') as h5_w:
        #             with h5py.File(
        #                     os.path.join(cfg.path_features, _seq,
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


def str2bool(v):
    return v.lower() in ('true', '1')

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
        '--pairwise_matching',
        default=False,
        action='store_true',
        help='Enable for pairwise matching methods')
    parser.add_argument(
        '--path_features',
        type=str,
        help='Path to the features to import')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['phototourism', 'googleurban', 'pragueparks'],
        help='List of datasets')
    parser.add_argument(
        '--tasks',
        default=['stereo','multiview'],
        help='List of tasks'
        )
    parser.add_argument(
        '--path_results',
        type=str,
        default='../benchmark-results',
        help='Directory holding benchmark results.')
    parser.add_argument(
        '--path_json',
        type=str,
        default='',
        help='Submission json file')
    parser.add_argument(
        '--subset',
        type=str,
        default='both',
        help='Subset to import: "val", "test", "both" (default), "spc-fix"')
    parser.add_argument(
        '--is_challenge',
        type=str2bool,
        default=False,
        help='Enable for challenge entries')

    cfg, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        parser.print_usage()
        exit(-1)

    if cfg.path_json == '':
        if cfg.is_challenge:
            raise RuntimeError('Must provide json file for challenge submission')
        if not cfg.kp_name:
            raise RuntimeError('Must define kp_name')
        if not cfg.desc_name:
            raise RuntimeError('Must define desc_name')
        if cfg.match_name and cfg.num_keypoints != -1:
            raise RuntimeError('Can not crop keypoints list with a custom matcher')

        cfg.kp_name = validate_label(cfg.kp_name)
        cfg.desc_name = validate_label(cfg.desc_name)
        cfg.match_name = validate_label(cfg.match_name)
    else:
        # read keypoints, descriptor, and match name from json
        method_list = load_json(cfg.path_json)
        if len(method_list)!=1:
            raise RuntimeError('Multiple method found in json file. Only support json fils with single method')
        cfg.method_dict = method_list[0]
        cfg.kp_name = cfg.method_dict['config_common']['keypoint']
        cfg.desc_name = cfg.method_dict['config_common']['descriptor']
        if cfg.method_dict['config_phototourism_stereo']['use_custom_matches']:
            cfg.match_name = {}
            for data_task in itertools.product(cfg.datasets, cfg.tasks):
                key = data_task[0]+'_'+data_task[1]
                cfg.match_name[key] = cfg.method_dict['config_'+key]['custom_matches_name']



    if cfg.is_challenge:
        path_tar_json = os.path.join(os.path.dirname(cfg.path_json),'formatted_'+os.path.basename(cfg.path_json))
        # compute hash for h5 files
        hash_str = hash_folder(cfg.path_features)
        # reformat json with proper incident
        reformat_json(cfg.path_json, path_tar_json)
        # add hash prefix to import path for challenge submissions
        cfg.path_results = os.path.join(cfg.path_results, 'challenge', hash_str)
        # add hash prefix to json label
        add_hash_prfix(cfg.path_json, path_tar_json, hash_str)


    print('Processing the following datasets: {} '.format(cfg.datasets))

    if isinstance(cfg.match_name,dict):
        match_name_str = 'custom match'
    elif cfg.match_name:
        match_name_str = cfg.match_name
    else:
        match_name_str = 'N/A'
    print('Importing {}, kp:"{}", desc="{}", matcher="{}", num_keypoints="{}" '.
          format(cfg.datasets,
                 cfg.kp_name, cfg.desc_name,
                 match_name_str,
                 cfg.num_keypoints if cfg.num_keypoints != -1 else 'N/A'))

    import_features(cfg)
