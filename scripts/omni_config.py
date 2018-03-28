import numpy as np
import pandas as pd
import os
import yaml
from kmerprediction import constants

directory = 'config_files/omni/'

if not os.path.exists(directory):
    os.makedirs(directory)

# Generate base config for omnilog data
base_yaml = {'augment': False,
             'augment_args': {},
             'scaler': 'scale_to_range',
             'scaler_args': {'high': 1, 'low': -1},
             'selection': 'select_k_best',
             'selection_args': {'k': 270, 'score_func': 'f_classif'},
             'data_method': 'get_omnilog_data',
             'data_args': {
                 'kwargs': {
                     'fasta_header': 'Strain',
                     'train_header': None,
                     'metadata': constants.OMNILOG_METADATA,
                     'one_vs_all': False,
                     'label_header': 'Host',
                     'prefix': '',
                     'suffix': ''
                 },
             },
             'model': 'neural_network',
             'model_args': {},
             'reps': snakemake.config['reps'],
             'validate': True
            }

with open(directory + 'neural_network_Host_omni_all.yml', 'w') as f:
    yaml.dump(base_yaml, f)

#Generate base config for kmer data
base_yaml['data_method'] = 'get_kmer'
base_yaml['data_args']['database'] = constants.DEFAULT_DB
base_yaml['data_args']['kwargs']['prefix'] = constants.OMNILOG_FASTA
base_yaml['data_args']['kwargs']['suffix'] = '.fasta'
base_yaml['data_args']['kwargs']['extra_header'] = 'WGS'
base_yaml['data_args']['kwargs']['extra_label'] = 1

with open(directory + 'neural_network_Host_kmer_all.yml', 'w') as f:
    yaml.dump(base_yaml, f)

yaml_files = [directory + x for x in os.listdir(directory) if '.yml' in x]

# generate base config for each model
models = ['random_forest', 'support_vector_machine']
for yf in yaml_files:
    with open(yf, 'r') as f:
        curr_yaml = yaml.load(f)
    for m in models:
        curr_yaml['model'] = m
        new_name = yf.replace('neural_network', m)
        with open(new_name, 'w') as f:
            yaml.dump(curr_yaml, f)

yaml_files = [directory + x for x in os.listdir(directory) if '.yml' in x]

# generate base config for each prediction
predictions = ['H type', 'O type', 'Serotype']
for yf in yaml_files:
    with open(yf, 'r') as f:
        curr_yaml = yaml.load(f)
    for p in predictions:
        curr_yaml['data_args']['kwargs']['label_header'] = p
        converted_prediction = ''.join(p.split(' '))
        new_name = yf.replace('Host', converted_prediction)
        with open(new_name, 'w') as f:
            yaml.dump(curr_yaml, f)

serotype_files = [directory + x for x in os.listdir(directory)
                  if '_Serotype_' in x]
otype_files = [directory + x for x in os.listdir(directory)
               if '_Otype_' in x]
htype_files = [directory + x for x in os.listdir(directory)
               if '_Htype_' in x]
host_files = [directory + x for x in os.listdir(directory)
              if '_Host_' in x]

complete_combos = [(serotype_files, constants.VALID_SEROTYPES),
                   (otype_files, constants.VALID_OTYPES),
                   (htype_files, constants.VALID_HTYPES),
                   (host_files, constants.VALID_HOSTS)]

for cc in complete_combos:
    filenames = cc[0]
    one_vs_alls = cc[1]
    for yf in filenames:
        with open(yf, 'r') as f:
            curr_yaml = yaml.load(f)
        for ova in one_vs_alls:
            curr_yaml['data_args']['kwargs']['one_vs_all'] = ova
            new_name = yf.replace('all', ova)
            with open(new_name, 'w') as f:
                yaml.dump(curr_yaml, f)
