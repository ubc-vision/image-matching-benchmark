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

def warn_or_error(cfg, msg, is_critical=False):

    err = False

    if cfg.error_level == 1 and is_critical:
        err = True
    if cfg.error_level == 2:
        err = True

    if not err:
        print('WARNING: {}'.format(msg))
    else:
        raise RuntimeError(msg)
