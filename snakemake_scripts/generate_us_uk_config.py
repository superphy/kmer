"""
Generate all of the config files necessary to perform the analysis from
the paper.
"""
import sys

sys.path.append('../../')

import os
import yaml
import constants

def generate_config_files():
    directory = '../config_files/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('base.yml', 'r') as f:
        base_yaml = yaml.load(f)

    base_yaml['reps'] = constants.DEFAULT_REPITITIONS

    models = ['support_vector_machine', 'random_forest', 'neural_network']
    data_methods = ['get_kmer_us_uk_split', 'get_kmer_us_uk_mixed',
                    'get_genome_region_us_uk_mixed', 'get_genome_region_us_uk_split']

    for m in models:
        base_yaml['model'] = m
        for dm in data_methods:
            base_yaml['data_method'] = dm
            dm = dm.split('_')
            file_name = directory + m + '_' + dm[1] + '_' + dm[-1] + '.yml'
            with open(file_name, 'w') as f:
                yaml.dump(base_yaml, f)

if __name__ == '__main__':
    generate_config_files()
