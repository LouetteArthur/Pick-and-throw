"""
A script for submitting multiple Experiments on the Alan cluster.
"""

# standard library
import argparse
import os
import yaml


BASE_SCRIPT = '''#!/usr/bin/env bash
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --job-name={job_name}         # Name of the job
#SBATCH --output={job_name}.log       # Log-file (important!)
#SBATCH --cpus-per-task={nb_cpu}      # Number of CPU cores to allocate
#SBATCH --mem-per-cpu={mem_per_cpu}G  # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:{nb_gpu}           # Number of GPU's
#SBATCH --time={max_time}             # Max execution time
#SBATCH --partition={partition}       # Partition to use for selecting GPU type

# make conda commands available
source ~/anaconda3/etc/profile.d/conda.sh

# activate the conda environment
conda activate Sim38

# run experiment
python3 train_agent.py {experiment_args}
'''

DEFAULT_SLURM_ARGS = {
    'nb_cpu': 4,
    'mem_per_cpu': 32,
    'nb_gpu': 0,
    'max_time': '7-00:00:00',
    'partition': 'all',
    'conda_env': 'base',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Submit a series of experiments to slurm.')
    parser.add_argument('config_file',
                        help='The path to the configuration yaml file.')
    return parser.parse_args()

def load_experiment_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  # Load the yaml file
    return config

def write_n_launch_slurm_script(series, job_name, slurm_args, experiment_args):
    # Launch the job from the slurm directory, so that the logfile is saved there
    slurm_dir = os.path.expanduser(os.environ.get('SLURM_DIR', '~/slurm_scripts'))
    os.makedirs(slurm_dir, exist_ok=True)  # Create the directory if it doesn't exist
    os.chdir(slurm_dir)
    job_file = f'{job_name}.sh'

    # remove existing files
    try:
        os.remove(job_file)
    except FileNotFoundError:
        pass
    try:
        os.remove(f'{job_name}.log')
    except FileNotFoundError:
        pass

    # create the slurm script
    with open(job_file, 'w') as fp:
        fp.write(BASE_SCRIPT.format(
            job_name=job_name,
            **slurm_args,
            experiment_args=' '.join(f'{k} {v}' for k, v in experiment_args.items()),
            series=series,
        ))
    os.system(f'sbatch {job_file}')

if __name__ == '__main__':
    args = parse_args()
    experiment_config = load_experiment_config(args.config_file)  # Load experiment config from yaml
    for experiment_name, experiment_args in experiment_config.items():
        write_n_launch_slurm_script(
            series='example_series',
            job_name=experiment_name,
            slurm_args=DEFAULT_SLURM_ARGS,
            experiment_args=experiment_args
        )