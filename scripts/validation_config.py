from kmerprediction import constants
import os
import yaml

base = {'augment': False,
        'augment_args': {},
        'data_args': {'kmer_kwargs': {}},
        'scaler': 'scale_to_range',
        'scaler_args': {'high': 1, 'low': -1},
        'validate': True,
        'verbose': True,
        'reps': snakemake.config['reps']}

data_methods = {'split': 'get_kmer_us_uk_split',
                'mixed': 'get_kmer_us_uk_mixed',
                'US': 'get_kmer_us',
                'UK': 'get_kmer_uk',
                'reverse_split': 'get_kmer_us_uk_reverse_split'}

selection_methods = {'kbest': 'select_k_best', 'fdr': 'select_fdr'}

selection_args = {'kbest': {'score_func': 'f_classif', 'k': 270},
                  'fdr': {'alpha': 1e-5, 'score_func': 'f_classif'}}

complete_path = '/home/rylan/Data/lupolova_data/complete_database/complete_'
complete_dbs = lambda x: complete_path + str(x) + '-mer_DB/'

output_path = '/home/rylan/Data/lupolova_data/database/'
output_dbs = lambda x: output_path + str(x) + '-mer_output_DB/'

filter_path = '/home/rylan/Data/lupolova_data/filtered_database/'
filter_dbs = lambda x: filter_path + str(x) + '-mer_output_DB/'

def main():
    directory = 'config_files/validation/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    model = snakemake.wildcards['model']
    k = int(snakemake.wildcards['k'])
    f = snakemake.wildcards['filter']
    dataset = snakemake.wildcards['dataset']
    selection = snakemake.wildcards['selection']

    if 'complete' in f:
        base['data_args']['complete_count'] = True
        base['data_args']['database'] = complete_dbs(k)
        base['data_args']['kmer_kwargs']['k'] = k
        base['data_args']['kmer_kwargs']['output_db'] = output_dbs(k)
        if 'filtered' in f:
            base['data_args']['kmer_kwargs']['name'] = 'minFileCount273'
            base['data_args']['kmer_kwargs']['min_file_count'] = 273
    elif 'filtered' in f:
        base['data_args']['complete_count'] = False
        base['data_args']['database'] = filter_dbs(k)
        base['data_args']['kmer_kwargs']['k'] = k
        limit = int(f.replace('filtered_', ''))
        base['data_args']['kmer_kwargs']['limit'] = limit

    base['model'] = model
    base['selection'] = selection_methods[selection]
    base['selection_args'] = selection_args[selection]

    with open(snakemake.output[0], 'w') as f:
        yaml.dump(base, f)

if __name__ == "__main__":
    main()
