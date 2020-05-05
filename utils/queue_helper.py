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

import hashlib
import os
import socket
import subprocess
import json
from time import sleep

from utils.feature_helper import is_feature_complete
from utils.match_helper import is_match_complete
from utils.filter_helper import is_filter_complete
from utils.stereo_helper import is_stereo_complete, is_geom_complete
from utils.err_helper import warn_or_error
from utils.path_helper import get_eval_path


def is_job_complete(mode, cfg):
    if mode == 'feature':
        return is_feature_complete(cfg)
    elif mode == 'match':
        return is_match_complete(cfg)
    elif mode == 'filter':
        return is_filter_complete(cfg)
    elif mode == 'model':
        return is_geom_complete(cfg)
    elif mode == 'stereo':
        return is_stereo_complete(cfg)
    else:
        raise ValueError('Unknown job type')


def create_job_key(mode, cfg):
    return get_eval_path(mode, cfg)


def get_cluster_name():

    hostname = socket.gethostname()

    if hostname.startswith('cedar') or hostname.startswith('cdr'):
        return 'cedar'
    elif hostname.startswith('gra'):
        return 'graham'
    elif hostname.startswith('beluga') or hostname.startswith('blg'):
        return 'beluga'
    elif hostname.startswith('nia'):
        return 'niagara'
    elif hostname.startswith('vps') or hostname.startswith('benchmark'):
        return 'gcp'
    elif hostname.startswith('n'):
        return 'rci'
    else:
        return 'other'


def write_sh_header(f, cfg):
    '''Writes headers for jobs to be submitted to slurm.'''
    f.write('#!/bin/bash\n')

    hostname = get_cluster_name()

    # No modules on GCP
    if hostname == 'gcp':
        return

    # Only create scripts with module changes when non-interactive
    if cfg.run_mode == 'batch':
        # Purge all modules -- this prevents race issues
        f.write('module purge\n')
        # Modules to be loaded
        if hostname == 'cedar' or hostname == 'graham':
            # If host is cedar/graham
            f.write('module load arch/avx2 StdEnv/2016.4\n')
            f.write('module load nixpkgs/16.09  gcc/5.4.0 colmap/3.5\n')
        elif hostname == 'beluga':
            # If host is beluga
            f.write('module load arch/avx2 StdEnv/2016.4\n')
            f.write('module load nixpkgs/16.09  gcc/5.4.0 colmap/3.5\n')
            f.write('module load hdf5\n')
        elif hostname == 'niagara':
            # If host is niagara
            f.write('module load CCEnv\n')
            f.write('module load arch/avx2 StdEnv/2016.4\n')
            f.write('module load nixpkgs/16.09  gcc/5.4.0 colmap/3.5 python\n')
        else:
            raise RuntimeError('Unknown server (batch mode)')
        # Load miniconda
        f.write('module load miniconda3\n')
        # Activate conda environment
        f.write('source activate {}\n'.format(cfg.conda_env))


def create_sh_cmd(script_name, cfg):
    '''Write's command to run the script with current configurations.'''

    # Write the script name
    cmd = 'python {}'.format(script_name)

    # Write all configurations that are not empty
    for key, value in cfg.__dict__.items():
        if len(str(value)) > 0:
            if isinstance(value, dict):
                cmd += ' --{}=\'{}\''.format(key, json.dumps(value))
            elif isinstance(value, list):
                value_str = ' '.join(str(v) for v in value)
                cmd += ' --{} {} '.format(key, value_str)
            else:
                cmd += ' --{}={}'.format(key, value)
    cmd = cmd.replace('\n', '')  # .replace(''', ''\''')

    # Terminate the line
    cmd += '\n'

    # print ('command is ',cmd)

    return cmd


def queue_job(job_file_fullpath, cfg, num_queue=1, dep_str=None, cpu=None):
    '''Get the command prefix depending on the server.

    Parameters
    ----------
    job_file_full_path: str
        Path to the job file to queue

    cfg: Namespace
        Configuration. If `cfg.run_mode` == "interactive", this function will
        simply run the job immediately rather than queueing.

    num_queue: int
        Number of times this job should be queued

    dep_str: str
        Any dependencies

    Returns
    -------
    job_id: str
        Returns the final job id that was queued as string

    '''

    if cfg.run_mode == 'interactive':
        com = ['bash', job_file_fullpath]
        interactive_res = subprocess.run(com)
        if interactive_res.returncode != 0:
            raise RuntimeError('Subprocess error!')
        job_id = None

    elif cfg.run_mode == 'batch':
        hostname = get_cluster_name()

        # Per-cluster settings (mostly CC)
        # Use a full node in any case
        nodes = 1
        if hostname == 'cedar':
            if cpu is None:
                cpu = 32
            mem = int(cpu * 128000 / 32)
            ntasks = None
            ntasks_per_node = None
        elif hostname == 'graham':
            if cpu is None:
                cpu = 32
            mem = int(cpu * 128000 / 32)
            ntasks = None
            ntasks_per_node = None
        elif hostname == 'beluga':
            if cpu is None:
                cpu = 40
            mem = 95000
            mem = int(cpu * 95000 / 40)
            nodes = 1
            ntasks = None
            ntasks_per_node = None
        elif hostname == 'niagara':
            cpu = None
            ntasks = 80
            ntasks_per_node = None
            mem = None
        elif hostname == 'gcp':
            cpu = None
            ntasks = 1
            # On a N-core node, the scheduler will allocate up to
            # N / ntasks_per_node simultaneous tasks with these settings
            ntasks_per_node = None
            mem = None
        else:
            raise RuntimeError('Unknown server (batch mode)')

        for _ in range(num_queue):
            # SLURM command
            com = ['sbatch']
            if cpu is not None:
                com += ['--cpus-per-task={}'.format(cpu)]
            if mem is not None:
                com += ['--mem={}'.format(mem)]
            if nodes is not None:
                com += ['--nodes={}'.format(nodes)]
            if ntasks is not None:
                com += ['--ntasks={}'.format(ntasks)]
            if ntasks_per_node is not None:
                com += ['--ntasks-per-node={}'.format(ntasks_per_node)]
            com += ['--time=0-{}'.format(cfg.cc_time)]
            # Account
            com += ['--account={}'.format(cfg.cc_account)]
            # Dependency
            if dep_str is not None:
                com += ['--dependency=afterany:{}'.format(dep_str)]
            # Output
            hostname = get_cluster_name()
            if hostname == 'gcp':
                com += [
                    '--output={}/{}.out'.format(
                        cfg.slurm_logs_path,
                        os.path.splitext(
                            os.path.basename(job_file_fullpath))[0])
                ]
            else:
                com += ['--output={}/%x-%j.out'.format(cfg.slurm_logs_path)]
            # # Do *NOT* carry on environments. This can cause issues.
            # com += ['--export=NONE']
            com += [job_file_fullpath]
            # Queue the job
            print(' '.join(com))
            slurm_res = subprocess.run(com, stdout=subprocess.PIPE)
            print(slurm_res.stdout.decode().rstrip('\n'))
            # Get job ID
            if slurm_res.returncode != 0:
                raise RuntimeError('Slurm error!')
            job_id = slurm_res.stdout.decode().split()[-1]
            dep_str = str(job_id)

            # On GCP, wait a little to avoid errors
            # (sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying)
            if hostname == 'gcp':
                sleep(0.25)
    else:
        raise ValueError('Wrong run mode of {}'.format(cfg.run_mode))

    return str(job_id)


def create_and_queue_jobs(cmd_list, cfg, dep_str=None, num_queue=1, cpu=None):
    '''
    Create jobs for feature extraction and queue them. Returns the final job
    id as string, to be used later for dependencies.
    '''

    # Check jobs directory and logs directory
    if not os.path.exists(cfg.slurm_jobs_path):
        os.makedirs(cfg.slurm_jobs_path)
    if not os.path.exists(cfg.slurm_logs_path):
        os.makedirs(cfg.slurm_logs_path)

    # Create the job file name using hashing
    job_file_fullpath = os.path.join(
        cfg.slurm_jobs_path,
        hashlib.sha256('_'.join(cmd_list).encode()).hexdigest())

    # Warn if job file already exists
    if os.path.exists(job_file_fullpath):
        warn_or_error(cfg, '{} already exists!'.format(job_file_fullpath))

    # Create job file
    with open(job_file_fullpath, 'w') as f:

        # Write the header
        write_sh_header(f, cfg)

        # Write the content
        job_cnt = 0
        for cmd in cmd_list:
            # This does not work, but we disabled it anyway for other issues
            # TODO  redirect output into sub-logs for cfg.parallel > 1
            # $ cmd > log.txt 2>&1 &
            # f.write(cmd.replace('\n', ' &> /dev/null &\n'))
            # f.write(cmd.replace('\n', ' &\n'))
            f.write(cmd)
            job_cnt += 1
            # Check if we need to wait
            # if job_cnt >= cfg.parallel:
            #     # Wait for currently running jobs to finish
            #     f.write('wait\n')
            #     job_cnt = 0

        # Wait for the processes to finish
        # if job_cnt > 0:
        #     f.write('wait\n')

    # Queue/run the job file
    return queue_job(job_file_fullpath,
                     cfg,
                     num_queue=num_queue,
                     dep_str=dep_str,
                     cpu=cpu)


def estimate_runtime(cfg_list):
    '''
    Returns the estimated runtime in hours. Used to split and schedule jobs
    with short expected walltime on systems which favour it such as CC.'''

    # Depending on hostname, set the multiplier.  Note that all default times
    # are measured on Niagara. This could be slower on Cedar, Graham, and
    # Beluga
    hostname = get_cluster_name()
    multiplier = 2
    # Estimate runtime on slurm
    if hostname == 'niagara':
        # 1.5 is still too strict
        multiplier = 2
    elif hostname == 'beluga':
        # This seems to result in jobs exceeding 3 hours
        # multiplier = 80. / 40
        multiplier = 3
    elif hostname == 'cedar':
        multiplier = 80. / 32
    elif hostname == 'graham':
        multiplier = 80. / 32
    elif hostname == 'gcp':
        # GCP runs one job at a time, but too many queued jobs break slurm
        # lower the multiplier to put more jobs on each colmap script
        # multiplier = 33
        multiplier = 10
    elif hostname == 'other':
        multiplier = 33
    else:
        NotImplementedError('Needs time measuring on {}'.format(hostname))

    est_time = 0
    for cfg in cfg_list:
        if cfg.bag_size == 3:
            # 3 bags
            est_time += 0.4 / 60
        elif cfg.bag_size == 5:
            # 5 bags
            est_time += 0.6 / 60
        elif cfg.bag_size == 10:
            # 10 bags
            est_time += 2. / 60
        elif cfg.bag_size == 25:
            # 25 bags
            est_time += 8. / 60
        else:
            # For all other jobs
            est_time += 30. / 60

    return est_time * multiplier
