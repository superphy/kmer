"""
Recreates the analysis from the paper
"""
import yaml
import os
import sys
sys.path.append('../../')
from run import main
from generate_config import generate_config_files

def run_all():
    directory = './config_files/'
    results_dir = './results/'

    if not os.exists(directory):
        generate_config_files()

    if not os.exists(results_dir):
        os.makedirs(results_dir)

    all_files = [x for x in os.listdir(directory) if '.yml' in x]

    for f in all_files:
        main(f, results_dir + '%s' % f, f.replace('.yml', ''))

if __name__ == '__main__':
    run_all()
