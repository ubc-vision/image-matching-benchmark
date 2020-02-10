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
import matplotlib
import matplotlib.pyplot as plt

def get_sequences(is_test):
    # ordered list of sequences
    if is_test:
        return {
            'british_museum': 'BM',
            'florence_cathedral_side': 'FCS',
            'lincoln_memorial_statue': 'LMS',
            'london_bridge': 'LB',
            'milan_cathedral': 'MC',
            'mount_rushmore': 'MR',
            'piazza_san_marco': 'PSM',
            'reichstag': 'RS',
            'sagrada_familia': 'SF',
            'st_pauls_cathedral': 'SPC',
            'united_states_capitol': 'USC',
        }
    else:
        return {
            'sacre_coeur': 'SC',
            'st_peters_square': 'SPS',
        }


def convert_bagsize_key(bagsize):
    if bagsize == 'bag3':
        return 'subset: 3 images'
    elif bagsize == 'bag5':
        return 'subset: 5 images'
    elif bagsize == 'bag10':
        return 'subset: 10 images'
    elif bagsize == 'bag25':
        return 'subset: 25 images'
    elif bagsize == 'bags_avg':
        return 'averaged over subsets'
    else:
        raise ValueError('Unknown bag size')


def parse_json(filename, verbose=False):
    with open(filename, 'r') as f:
        if verbose:
            print('Parsing "{}"'.format(filename))
        return json.load(f)


def get_plot_defaults():
    return {
        'font_size_title': 32,
        'font_size_axes': 25,
        'font_size_ticks': 22,
        'font_size_legend': 22,
        'line_width': 3,
        'marker_size': 10,
        'dpi': 600,
    }

def make_like_colab(fig, ax):
    # use a gray background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#E6E6E6')
    
    # draw solid white grid lines
    ax.grid(color='white', linestyle='solid')

    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # hide top and right ticks
    # ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()

    # lighten ticks and labels
    # ax.tick_params(colors='gray', direction='out')
    # for tick in ax.get_xticklabels():
    #     tick.set_color('gray')
    # for tick in ax.get_yticklabels():
    #     tick.set_color('gray')
    
    # remove the notch on the ticks (but show the label)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    
def color_picker_features(label):
    cmap = matplotlib.cm.get_cmap('tab20')
    
    # list position determines color: move them around if there are conflicts between similar colors
    # the legend will be sorted alphabetically
    known = [
        'CV-SIFT',
        'CV-RootSIFT',
        'CV-AKAZE',
        'CV-ORB',
        'CV-SURF',
        'VL-DoG-SIFT',
        'VL-DoGAff-SIFT',
        'VL-Hess-SIFT',
        'VL-HessAffNet-SIFT',
        'ContextDesc',
        'D2-Net (SS)',
        'D2-Net (MS)',
        'SuperPoint',
        'LF-Net',
        'CV-DOG/LogPolarDesc',
        'CV-DoG/SOSNet',
        'CV-DoG/GeoDesc',
        'CV-DoG/HardNet',
        'KeyNet/HardNet',
        'L2Net',
        'DSP-SIFT',
        'Colmap-SIFT',
        
        
    ]
    if label == 'Key.Net/HardNet':
        label = 'KeyNet/HardNet'
    for i, s in enumerate(known):
        if s.lower() in label.lower():
            if i >=20:
#                 return 'black'
                return (.15, .15, .15, 1)
            else:
                return cmap(i)
    
    raise ValueError('Could not find color for method "{}"- > please see utils.py'.format(label))
    
#     if len(known
