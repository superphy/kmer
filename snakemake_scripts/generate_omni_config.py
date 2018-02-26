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

    base_yaml['data_args']['kwargs']['metadata'] = constants.OMNILOG_METADATA
    base_yaml['reps'] = constants.DEFAULT_REPITITIONS

    base_yaml['data_args']['kwargs']['one_vs_all'] = False
    base_yaml['data_args']['kwargs']['label_header'] = 'Host'
    base_yaml['model'] = 'neural_network'
    base_yaml['data_args']['kwargs']['remove'] = 'Unknown'

# create base omni file
    base_yaml['data_method'] = 'get_omnilog_data'
    base_yaml['data_args']['database'] = None
    base_yaml['data_args']['kwargs']['prefix'] = ''
    base_yaml['data_args']['kwargs']['suffix'] = ''

    with open(directory + 'neural_network_Host_omni_all.yml', 'w') as f:
        yaml.dump(base_yaml, f)

# create base kmer file
    base_yaml['data_method'] = 'get_kmer'
    base_yaml['data_args']['database'] = constants.DB
    base_yaml['data_args']['kwargs']['prefix'] = constants.OMNILOG_FASTA
    base_yaml['data_args']['kwargs']['suffix'] = '.fasta'

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
    predictions = [('H type','?'),  ('O type','OR'), ('Serotype','O111:?')]
    for yf in yaml_files:
        with open(yf, 'r') as f:
            curr_yaml = yaml.load(f)
        for p in predictions:
            curr_yaml['data_args']['kwargs']['label_header'] = p[0]
            curr_yaml['data_args']['kwargs']['Remove'] = p[1]
            converted_prediction = ''.join(p[0].split(' '))
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

    serotypes = ['O103:H2', 'O113:H21', 'O121:H19', 'O145:NM', 'O157:H7',
                 'O26:H11', 'O45:H2', 'O91:H21']
    otypes = ['O103', 'O104', 'O111', 'O113', 'O121', 'O145', 'O157', 'O26',
              'O45', 'O91']
    htypes = ['H11', 'H19', 'H21', 'H2', 'H7', 'NM']
    hosts = ['Human', 'Bovine', 'Water']

    complete_combos = [(serotype_files, serotypes), (otype_files, otypes),
                       (htype_files, htypes), (host_files, hosts)]

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

if __name__ == '__main__':
    generate_config_files()
