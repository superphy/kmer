from kmerprediction import constants
import os
import yaml

directory = 'config_files/US_UK/'

if not os.path.exists(directory):
    os.makedirs(directory)

base_yaml = {'augment': False,
             'augment_args': {},
             'data_args': {'kmer_kwargs': {}},
             'scaler': 'scale_to_range',
             'scaler_args': {'high': 1, 'low': -1},
             'selection': 'f_test_threshold',
             'selection_args': {'threshold': 0.2},
             'validate': True,
             'verbose': True,
             'reps': snakemake.config['reps']}

models = ['support_vector_machine', 'random_forest', 'neural_network']
data_methods = ['get_kmer_us_uk_split', 'get_kmer_us_uk_mixed']
selection_methods = [('kbest', 'select_k_best', {'score_func': 'f_classif', 'k': 270}),
                     ('pthreshold', 'f_test_threshold', {'threshold': 0.2})]

base_path = '/home/rylan/miniconda3/envs/kmer/lib/python3.6/site-packages/kmerprediction/kmer_data/'
complete_dbs = {7: base_path + 'complete_7-mer_DB/',
                15: base_path + 'complete_15-mer_DB/',
                31: base_path + 'complete_31-mer_DB/'}
base_path = '/home/rylan/Data/lupolova_data/database/'
output_dbs = {7: base_path + '7-mer_output_DB/',
              15: base_path + '15-mer_output_DB/',
              31: base_path + '31-mer_output_DB/'}

for m in models:
    base_yaml['model'] = m
    for dm in data_methods:
        base_yaml['data_method'] = dm
        dm = dm.split('_')
        for k in snakemake.config['k_vals']:
            base_yaml['data_args']['kmer_kwargs']['k'] = k
            base_yaml['data_args']['database'] = complete_dbs[k]
            base_yaml['data_args']['kmer_kwargs']['output_db'] = output_dbs[k]
            for key, selection_method, selection_args in selection_methods:
                base_yaml['selection'] = selection_method
                base_yaml['selection_args'] = selection_args
                k = str(k)
                file_name = directory+m+'_'+dm[1]+'_'+dm[-1]+'_'+k+'_'+key+'.yml'
                with open(file_name, 'w') as f:
                    yaml.dump(base_yaml, f)
