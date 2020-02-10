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

import pkgutil
from importlib import import_module


# import torch and check if gpu available
try:
    torch = import_module('torch')
    gpu_found = torch.cuda.is_available()
except Exception:
    gpu_found = False

# Import all modules within this folder. Note that this does not recurse down.
__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Skip GPU modules
    # ...

    # Skip scikit-image modules if not available
    if 'skimage' in module_name:
        scikit_available = True
        try:
            import skimage
        except Exception:
            scikit_available = False
        if not scikit_available:
            continue
    if 'intel' not in module_name:    
        __all__.append(module_name)
        _module = loader.find_module(module_name).load_module(module_name)
        globals()[module_name] = _module
