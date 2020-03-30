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
from glob import glob
from collections import OrderedDict
from copy import deepcopy
from time import time
from config import get_config, print_usage
from utils import pack_helper
from utils.io_helper import load_h5, load_json
from utils.path_helper import get_desc_file
from IPython import embed


if __name__ == '__main__':
    # Defaults
    input_path = './challenge-results'
    output_path = './challenge-results-new'
    bag_size_str = ['{}bag'.format(b) for b in [5, 10, 25]]

    # Get results
    files = glob('{}/*.json'.format(input_path))
    print('Found {} results'.format(len(files)))

    for i, _file in enumerate(files):
        cur_fn = os.path.basename(_file)
        print('Processing "{}" ({}/{})'.format(cur_fn, i + 1, len(files)))
        results = load_json(_file)

        del results['phototourism']['results']['united_states_capitol']

        res_dict = results['phototourism']['results']
        res_dict['allseq'] = OrderedDict()
        for task in ['stereo', 'multiview', 'relocalization']:
            res_dict['allseq'][task] = OrderedDict()
            res_dict['allseq'][task]['run_avg'] = OrderedDict()
            if task == 'multiview':
                for bag in bag_size_str + ['bag_avg']:
                    res_dict['allseq']['multiview']['run_avg'][
                        bag] = OrderedDict()

        # Compute average across scenes, for stereo
        t_cur = time()
        num_runs = results['phototourism']['num_runs_stereo']
        pack_helper.average_stereo_over_scenes(None, res_dict, num_runs)
        print(' -- Repacking stereo [{:.02f} s]'.format(time() - t_cur))

        # Compute average across scenes, for multiview
        t_cur = time()
        num_runs = results['phototourism']['num_runs_multiview']
        pack_helper.average_multiview_over_scenes(
            None,
            res_dict,
            num_runs,
            ['bag_avg'] + bag_size_str)
        print(' -- Repacking multiview [{:.02f} s]'.format(time() - t_cur))

        print(' -- Saving')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        json_dump_file = os.path.join(output_path, cur_fn)
        with open(json_dump_file, 'w') as outfile:
            json.dump(results, outfile, indent=4)
