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


# IMW 2021 Submission Validator
#
# Submission Zip file should have folder structure as follow:
# ├── config.json
# ├── [Dataset 1]
# │   ├── [Sequence 1]
# │   │   ├── keypoints.h5
# │   │   ├── descriptors.h5
# │   │   ├── matches.h5
# │   ├── [Sequence 2]
# │   │   ├── ...
# ├── [Dataset 2]
# │   ├── ...
# 
# In the config file, please following these nameing conventions:
# 'keypoint', 'descriptor', and 'custom_matches_name' should only 
# contains lowercase letters(a-z), numbers(0-9), and two special 
# charactors('-','.')
# 'json_label' should only contain above mentioned charactors plus '_'
#
# Please use this script to validate your zip file before submiting.
# This script will create a log file alongside with your submission file.
# Please make sure there is no error message in the log file.

import os
import sys
import argparse
from itertools import product 
from utils.io_helper import load_h5, load_json
from config import validate_method
from utils.pack_helper import get_descriptor_properties

def get_config():
    parser = argparse.ArgumentParser()
    # submission zip file path
    parser.add_argument('--submit_file_path', type=str, default='')
    # benchmark repo path
    parser.add_argument('--benchmark_repo_path', type=str, default='./')
    # dataset path
    parser.add_argument('--raw_data_path', type=str, default='../imw_data')
    # list of datasets
    parser.add_argument('--datasets', nargs='+', default=['phototourism', 'googleurban', 'pragueparks'])
    config = parser.parse_args()
    return config

class MonitorLogger():
	def __init__(self, logger_path, value):
		self.file = os.path.join(logger_path,'{}_log.txt'.format(value))
		if os.path.isfile(self.file):
			os.remove(self.file)

	def add_new_log(self, new_log):
		with open(self.file, 'a') as f:
			f.write(new_log + '\n')
	def is_empty(self):
		if os.path.isfile(self.file):
			return False
		else:
			return True
	def get_log_str(self):
		with open(self.file, 'r') as f:
			lines = f.readlines()
		return ''.join(lines)	
	def get_file_path(self):
		return self.file

def validate_submission_files(sub_path,benchmark_repo_path, datasets, raw_data_path, logger):
	for dataset in datasets:

		raw_dataset_path = os.path.join(raw_data_path,dataset)
		# check if dataset folder exists
		sub_dataset_path = os.path.join(sub_path,dataset)
		if not os.path.isdir(sub_dataset_path):
			logger.add_new_log('Submission does not contain {} dataset (ignore this mesage if you do not intend to evaluate on this dataset).'.format(dataset))
			continue
		# read seqs from json
		seqs = load_json(os.path.join(benchmark_repo_path,'json/data/{}_test.json'.format(dataset)))
		for seq in seqs:
			# get number of image
			raw_seq_path = os.path.join(raw_dataset_path,seq)
			im_list = [os.path.splitext(f)[0] for f in os.listdir(raw_seq_path) if (os.path.isfile(os.path.join(raw_seq_path, f)) and f.endswith(('png', 'jpg')))]
			num_im =len(im_list)

			# get all key pairs
			key_pairs = [pair[0]+'-'+pair[1] for pair in list(product(im_list, im_list))if pair[0] > pair[1]]

			# check if seq folder exists
			sub_seq_path = os.path.join(sub_dataset_path,seq)
			if not os.path.isdir(sub_seq_path):
				logger.add_new_log('Submission does not contain {} sequence in {}  dataset.'.format(seq,dataset))
				continue
			# validate keypoints file
			kp_path = os.path.join(sub_seq_path,'keypoints.h5')
			if not os.path.isfile(kp_path):
				logger.add_new_log('Submission does not contain keypoints file for {} sequence in {} dataset.'.format(seq,dataset))
			else:
				keypoints = load_h5(kp_path)
				if len(keypoints.keys()) == 0:
					logger.add_new_log('{}-{}: Keypoints file is corrupted'.format(dataset,seq))
				else:
					if sorted(list(keypoints.keys()))!=sorted(im_list):
						logger.add_new_log('{}-{}: Keypoints file does not contain all the image keys.'.format(dataset,seq))
					if len(list(keypoints.values())[0].shape)!=2:
						logger.add_new_log('{}-{}: Keypoints file is in wrong format.'.format(dataset,seq))
					if list(keypoints.values())[0].shape[1]!=2:
						logger.add_new_log('{}-{}: Keypoints file is in wrong format.'.format(dataset,seq))
					# check number of keypoints
					for _keypoints in keypoints.values():
						if _keypoints.shape[0] > 8000:
							logger.add_new_log('{}-{}: Keypoints file contains more than 8000 points.'.format(dataset,seq))
							break
			
			# check if match file exists first
			match_files = [file for file in os.listdir(sub_seq_path) if os.path.isfile(os.path.join(sub_seq_path,file)) and file.startswith('match')]	
			
			# validate descriptor file
			desc_path = os.path.join(sub_seq_path,'descriptors.h5')

			# much provide either descriptor file or match file 
			if not os.path.isfile(desc_path) and len(match_files)==0:
				logger.add_new_log('Submission does not contain descriptors file for {} sequence in {}  dataset.'.format(seq,dataset))
			elif not os.path.isfile(desc_path):
				pass
			else:
				descriptors = load_h5(desc_path)
				if len(descriptors.keys()) == 0:
					logger.add_new_log('{}-{}: Descriptors file is corrupted'.format(dataset,seq))
				else:
					if sorted(list(descriptors.keys()))!=sorted(im_list):
						logger.add_new_log('{}-{}: Descriptors file does not contain all the image keys.'.format(dataset,seq))
					if len(list(descriptors.values())[0].shape)!=2:
						logger.add_new_log('{}-{}: Descriptors file is in wrong format'.format(dataset,seq))
					if list(descriptors.values())[0].shape[1]<64 or list(descriptors.values())[0].shape[1]>2048:
						logger.add_new_log('{}-{}: Descriptors file is in wrong format'.format(dataset,seq))
					
					# check descriptor size
					desc_type, desc_size, desc_nbytes = get_descriptor_properties({},descriptors)
					if desc_nbytes > 512 and len(match_files)==0:
						logger.add_new_log('{}-{}: Descriptors size is larger than 512 bytes, you need to provide custom match file'.format(dataset,seq))

			# validate match file
			# check match file name
			if 'matches.h5' in match_files:
				if len(match_files) != 1:
					logger.add_new_log('{}-{}: matches.h5 exists. Do not need to provide any other match files.'.format(dataset,seq))
			elif 'matches_multiview.h5' in match_files or 'matches_stereo_0.h5' in match_files or 'matches_stereo.h5' in match_files:
				if 'matches_multiview.h5' not in match_files:
					logger.add_new_log('{}-{}: missing matches_multiview.h5'.format(dataset,seq))
				if 'matches_stereo_0.h5' not in match_files and 'matches_stereo.h5' not in match_files:
					logger.add_new_log('{}-{}: missing matches_stereo.h5'.format(dataset,seq))
				if 'matches_stereo_1.h5' in match_files or 'matches_stereo_2.h5' in match_files:
					logger.add_new_log('{}-{}: for 2021 challenge, we only run stereo once, no need to provide matches_stereo_1 and matches_stereo_2'.format(dataset,seq))

			for match_file in match_files:
				matches = load_h5(os.path.join(sub_seq_path,match_file))
				if len(matches.keys()) == 0:
					logger.add_new_log('{}-{}: Matches file is corrupted'.format(dataset,seq))
				else:
					if len(matches.keys()) != len(key_pairs):
						logger.add_new_log('{}-{}: Matches file contains wrong number of keys, should have {} keys, have {}.'.format(dataset,seq, len(key_pairs), len(matches.keys())))
					elif sorted(list(matches.keys()))!=sorted(key_pairs):
						logger.add_new_log('{}-{}: Matches file contains worng keys, maybe the image names is in reverse order. Plase refer to submission instruction for proper custom match key naming convention'.format(dataset,seq))
					if len(list(matches.values())[0].shape)!=2:
						logger.add_new_log('{}-{}: Matches file is in wrong format.'.format(dataset,seq))
					if list(matches.values())[0].shape[0]!=2:
						logger.add_new_log('{}-{}: Matches file is in wrong format.'.format(dataset,seq))	


def validate_json(json_path, datasets, logger):
	# check if json file exist
	if not os.path.isfile(json_path):
		logger.add_new_log('Submission does not contain json file')
		return
	# load json
	try:
		method_list = load_json(json_path)
	except:
		logger.add_new_log('Following error occurs when loading json : \n   {}'.format(sys.exc_info()))
		return

	# validate json
	if not type(method_list) is list:
		logger.add_new_log('Json should contain a list of method, please refer to the example json file.')
		return

	for i, method in enumerate(method_list):
		print('Validating method {}/{}: "{}"'.format(
		i + 1, len(method_list), method['config_common']['json_label']))
		try:
			validate_method(method, is_challenge=True, datasets=datasets)
		except:
			logger.add_new_log('Following error occurs when validating json : \n   {}'.format(sys.exc_info()))


def main():

	config = get_config()
	
	# Unzip folder
	submission_name = os.path.basename(config.submit_file_path).split('.')[0]
	extracted_folder = '{}_extracted'.format(submission_name)
	folder_path = os.path.dirname(config.submit_file_path)
	os.system('unzip {} -d {}'.format(config.submit_file_path,os.path.join(folder_path,extracted_folder)))

	# Init Logger
	logger = MonitorLogger(folder_path, submission_name)

	# Validate Submission files
	validate_submission_files(os.path.join(folder_path,extracted_folder), config.benchmark_repo_path, config.datasets, config.raw_data_path,logger)


	# Validate Json
	validate_json(os.path.join(folder_path,extracted_folder,'config.json'), config.datasets, logger)

	if logger.is_empty():
		logger.add_new_log('Submission is in proper format, please submit to IMW 2021 website.')
		print('--------\nSubmission is in proper format, please submit to IMW 2021 website.\n--------')
	else:
		logger.add_new_log('Please fix the above errors and rerun this script!')
		print('--------\nPlease fix the errors in log file before submitting!\n{}\n--------'.format(logger.get_file_path()))


if __name__ == "__main__":
    main()
