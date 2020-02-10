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
from utils.path_helper import get_filter_match_file


def is_filter_complete(cfg):
    '''Checks if stereo evaluation is complete.'''

    # We should have the colmap pose file and no colmap temp path
    is_complete = os.path.exists(get_filter_match_file(cfg))

    return is_complete
