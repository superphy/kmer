import numpy as np
import pandas as pd
import os
import yaml
from kmerprediction import constants

selection_methods = {'kbest': 'select_k_best'}
selection_args = {'kbest': {'score_func': 'f_classif', 'k': 270}}

predictions = {'Host': 'Host', 'Htype': 'H type',
               'Otype': 'O type', 'Serotype': 'Serotype'}

complete_path = '/home/rylan/Data/omnilog_data/complete_database/complete_'
complete_dbs = lambda x: complete_path + str(x) + '-mer_DB/'

output_path = '/home/rylan/Data/omnilog_data/output_database/'
output_dbs = lambda x: output_path + str(x) + '-mer_DB/'

filter_path = '/home/rylan/Data/omnilog_data/filter_database/'
filter_dbs = lambda x: filter_path + str(x) + '-mer_DB/'

base = {'augment': False,
        'augment_args': {},
        'validate': True,
        'scaler': 'scale_to_range',
        'reps': snakemake.config['reps'],
        'scaler_args': {'high': 1, 'low': -1},
        'data_args': {
            'metadata_kwargs': {'metadata': constants.OMNILOG_METADATA,
                                'fasta_header': 'Strain',
                                'train_header': None},
        }
       }

def main():
    model = snakemake.wildcards['model']
    selection = snakemake.wildcards['selection']
    prediction = snakemake.wildcards['prediction']
    one_vs_all = snakemake.wildcards['ova']

    base['model'] = model
    base['selection'] = selection_methods[selection]
    base['selection_args'] = selection_args[selection]

    base['data_args']['metadata_kwargs']['label_header'] = predictions[prediction]
    if one_vs_all == 'all':
        base['data_args']['metadata_kwargs']['one_vs_all'] = False
    else:
        base['data_args']['metadata_kwargs']['one_vs_all'] = one_vs_all

    wc = snakemake.wildcards.keys()
    if 'k' in wc and 'filter' in wc:
        k = int(snakemake.wildcards['k'])
        f = snakemake.wildcards['filter']
        base['data_args']['kmer_kwargs'] = {}
        if 'complete' in f:
            base['data_args']['complete_count'] = True
            base['data_args']['database'] = complete_dbs(k)
            base['data_args']['kmer_kwargs']['output_db'] = output_dbs(k)
            if 'filtered' in f:
                base['data_args']['kmer_kwargs']['name'] = 'min143'
                base['data_args']['kmer_kwargs']['min_file_count'] = 143
        elif 'filtered' in f:
            base['data_args']['complete_count'] = False
            base['data_args']['database'] = filtered_dbs(k)
        base['data_method'] = 'get_kmer'
        base['data_args']['metadata_kwargs']['prefix'] = constants.OMNILOG_FASTA
        base['data_args']['metadata_kwargs']['suffix'] = '.fasta'
        base['data_args']['metadata_kwargs']['extra_header'] = 'WGS'
        base['data_args']['metadata_kwargs']['extra_label'] = 1
        base['data_args']['kmer_kwargs']['k'] = k
    else:
        base['data_method'] = 'get_omnilog_data'
        base['data_args']['metadata_kwargs']['prefix'] = ''
        base['data_args']['metadata_kwargs']['suffix'] = ''

    with open(snakemake.output[0], 'w') as f:
        yaml.dump(base, f)

if __name__ == "__main__":
    main()
