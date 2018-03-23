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
             'selection_args': {'threshold': 0.2},
             'validate': True,
             'verbose': True,
             'reps': constants.DEFAULT_REPITITIONS}

models = ['support_vector_machine', 'random_forest', 'neural_network']
data_methods = ['get_kmer_us_uk_split', 'get_kmer_us_uk_mixed']
selection_methods = [('kbest', 'select_k_best', {'threshold': 0.2}),
                     ('pthreshold', 'f_test_threshold', {'score_func': 'f_classif', 'k': 270})]

for m in models:
    base_yaml['model'] = m
    for dm in data_methods:
        base_yaml['data_method'] = dm
        dm = dm.split('_')
        for k in snakemake.config['k_vals']:
            base_yaml['data_args']['k'] = k
            for key, selection_method, selection_args in selection_methods:
                base_yaml['selection'] = selection_method
                base_yaml['selection_args'] = selection_args
                k = str(k)
                file_name = directory+m+'_'+dm[1]+'_'+dm[-1]+'_'+k+'_'+key+'.yml'
                with open(file_name, 'w') as f:
                    yaml.dump(base_yaml, f)
