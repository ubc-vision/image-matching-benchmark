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

# This script should be sourced to prepare CC envs
module purge
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == "cedar"* ]] || [[ "$HOSTNAME" == "cdr"* ]] ; then
    module load arch/avx2 StdEnv/2016.4
    module load nixpkgs/16.09 gcc/5.4.0 colmap/3.5
elif [[ "$HOSTNAME" == "gra"* ]] ; then
    module load arch/avx2 StdEnv/2016.4
    module load nixpkgs/16.09 gcc/5.4.0 colmap/3.5
elif [[ "$HOSTNAME" == "mishkdmy"* ]] ; then
    module load COLMAP/3.6-dev.2-fosscuda-2019a
elif [[ "$HOSTNAME" == "nia"* ]] ; then
    module load CCEnv
    module load arch/avx2 StdEnv/2016.4
    module load nixpkgs/16.09 gcc/5.4.0 colmap/3.5 python
elif [[ "$HOSTNAME" == "beluga"* ]] || [[ "$HOSTNAME" == "blg"* ]]; then
    module load arch/avx2 StdEnv/2016.4
    module load nixpkgs/16.09 gcc/5.4.0 colmap/3.5
    module load hdf5
else
		echo "WARNING: Nothing loaded?"
fi

module load miniconda3



