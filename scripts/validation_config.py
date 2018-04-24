from kmerprediction import constants
import os
import yaml

base = {'augment': False,
        'augment_args': {},
        'scaler': 'scale_to_range',
        'scaler_args': {'high': 1, 'low': -1},
        'validate': True,
        'verbose': True,
        'reps': snakemake.config['reps']}

kmer_data_methods = {'split': 'get_kmer_us_uk_split',
                'mixed': 'get_kmer_us_uk_mixed',
                'US': 'get_kmer_us',
                'UK': 'get_kmer_uk',
                'reverse_split': 'get_kmer_us_uk_reverse_split'}

genome_data_methods = {'split': 'get_genome_region_us_uk_split',
                       'mixed': 'get_genome_region_us_uk_mixed',
                       'US': 'get_genome_region_us',
                       'UK': 'get_genome_region_uk',
                       'reverse_split': 'get_genome_region_us_uk_reverse_split'}

selection_methods = {'kbest': 'select_k_best', 'fdr': 'select_fdr',
                     'kbest197': 'select_k_best'}

selection_args = {'kbest': {'score_func': 'f_classif', 'k': 270},
                  'fdr': {'alpha': 1e-5, 'score_func': 'f_classif'},
                  'kbest197': {'score_func': 'f_classif', 'k': 197}}

complete_path = '/home/rylan/Data/lupolova_data/complete_database/complete_'
complete_dbs = lambda x: complete_path + str(x) + '-mer_DB/'

output_path = '/home/rylan/Data/lupolova_data/database/'
output_dbs = lambda x: output_path + str(x) + '-mer_output_DB/'

filter_path = '/home/rylan/Data/lupolova_data/filtered_database/'
filter_dbs = lambda x: filter_path + str(x) + '-mer_output_DB/'

tables = lambda x: constants.GENOME_REGIONS + 'binary_table_' + str(x) + '.txt'

def main():
    directory = 'config_files/validation/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    model = snakemake.wildcards['model']
    dataset = snakemake.wildcards['dataset']
    selection = snakemake.wildcards['selection']
    wc = snakemake.wildcards.keys()

    base['data_args'] = {}

    if 'k' in wc and 'filter' in wc:
        k = int(snakemake.wildcards['k'])
        f = snakemake.wildcards['filter']
        base['data_args']['kmer_kwargs'] = {}
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
        base['data_method'] = kmer_data_methods[dataset]
    elif 'fragment' in wc:
        fragment_size = snakemake.wildcards['fragment']
        base['data_args']['table'] = tables(fragment_size)
        base['data_method'] = genome_data_methods[dataset]

    base['model'] = model
    base['selection'] = selection_methods[selection]
    base['selection_args'] = selection_args[selection]

    with open(snakemake.output[0], 'w') as f:
        yaml.dump(base, f)

if __name__ == "__main__":
    main()
