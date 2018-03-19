from kmerprediction import constants
import os
import yaml

directory = 'config_files/US_UK/'

if not os.path.exists(directory):
    os.makedirs(directory)

base_yaml = {'augment': False,
             'augment_args': {},
             'data_args': {'directory': constants.OUTPUT_DIR},
             'scaler': 'scale_to_range',
             'scaler_args': {'high': 1, 'low': -1},
             'selection': 'f_test_threshold',
             'selection_args': {'threshold': 0.05},
             'validate': True,
             'reps': constants.DEFAULT_REPITITIONS}

models = ['support_vector_machine', 'random_forest', 'neural_network']
data_methods = ['get_kmer_us_uk_split', 'get_kmer_us_uk_mixed']

for m in models:
    base_yaml['model'] = m
    for dm in data_methods:
        base_yaml['data_method'] = dm
        dm = dm.split('_')
        for k in snakemake.config['k_vals']:
            base_yaml['data_args']['k'] = k
            for N in snakemake.config['N_vals']:
                base_yaml['data_args']['L'] = N
                k = str(k)
                N = str(N)
                file_name = directory+m+'_'+dm[1]+'_'+dm[-1]+'_'+k+'_'+N+'.yml'
                with open(file_name, 'w') as f:
                    yaml.dump(base_yaml, f)
