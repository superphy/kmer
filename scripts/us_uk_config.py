from kmerprediction import constants
import os
import yaml

directory = '../config_files/US_UK/'

if not os.path.exists(directory):
    os.makedirs(directory)

base_yaml = {'augment': False,
             'augment_args': {},
             'data_args': {},
             'scaler': 'scale_to_range',
             'scaler_args': {'high': 1, 'low': -1},
             'selection': 'select_k_best',
             'selection_args': {'k': 270, 'score_func': 'f_classif'},
             'validate': True,
             'reps': constants.DEFAULT_REPITITIONS}

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
