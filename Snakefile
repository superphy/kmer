import os
from kmerprediction import constants

ova_dict = {'Host': constants.VALID_HOSTS,
            'Htype': constants.VALID_HTYPES,
            'Otype': constants.VALID_OTYPES,
            'Serotypes': constants.VALID_SEROTYPES}

rule generate_us_uk_config:
    output:
        expand('config_files/US_UK/{model}_{data}_{split}.yml',
               model=['neural_network', 'support_vector_machine', 'random_forest'],
               data=['kmer', 'genome'],
               split=['split', 'mixed'])
    script:
        'scripts/us_uk_config.py'


rule generate_omni_config:
    output:
        expand('config_files/omni/{model}_Host_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_HOSTS + ['all']),
        expand('config_files/omni/{model}_Serotype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_SEROTYPES + ['all']),
        expand('config_files/omni/{model}_Otype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_OTYPES + ['all']),
        expand('config_files/omni/{model}_Htype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_HTYPES + ['all'])
    script:
        'scripts/omni_config.py'


rule run_single:
    input:
        'config_files/{dir}/{analysis}.yml'
    output:
        'results/{dir}/yaml/{analysis}.yml'
    script:
        'scripts/run_single.py'


rule make_omni_binary_data_frames:
    input:
        expand('results/omni/yaml/{model}_{{prediction}}_{data}_{ova}.yml',
               model=['support_vector_machine', 'neural_network', 'randomf_forest'],
               data=['kmer', 'omni'],
               ova=lambda wildcards: ova_dict[wildcards.prediction])
    output:
        'results/omni/DataFrames/{prediction}.csv'
    script:
        'scripts/make_binary_data_frames.py'


rule make_omni_multiclass_data_frames:
    input:
        expand('results/omni/yaml/{model}_{prediction}_{datatype}_all.yml',
               model=['random_forest', 'support_vector_machine', 'neural_network'],
               datatype=['kmer', 'omni'],
               prediction=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'results/omni/DataFrames/Multiclass.csv'
    script:
        'scripts/make_multiclss_data_frames.py'


rule plot_binary_omni:
    input:
        'results/omni/DataFrames/{prediction}.csv'
    output:
        'results/omni/Figures/{prediction}.pdf'
    script:
        'scripts/plot_binary.py'


rule plot_multiclass_omni:
    input:
        'results/omni/DataFrames/Multiclass.csv'
    output:
        'results/omni/Figures/Multiclass.pdf'
    script:
        'scripts/plot_multiclass.py'


rule make_all_omni_plots:
    input:
        expand('results/omni/Figures/{figure}.pdf',
               figure=['Multiclass', 'Host', 'Htype', 'Otype', 'Serotype'])


rule important_features_data_frame:
    input:
        expand('results/omni/yaml/{model}_{{prediction}}_omni_{{ova}}.yml',
               model=['random_forest', 'support_vector_machine'])
    output:
        'results/omni/Important_Features/{prediction}_{ova}.csv'
    script:
        'important_features.py'


rule all_important_features:
    input:
        expand('results/omni/Important_Features/Host_{ova}.csv',
               ova=constants.VALID_HOSTS + ['all']),
        expand('results/omni/Important_Features/Serotype_{ova}.csv',
               ova=constants.VALID_SEROTYPES + ['all']),
        expand('results/omni/Important_Features/Otype_{ova}.csv',
               ova=constants.VALID_OTYPES + ['all']),
        expand('results/omni/Important_Features/Htype_{ova}.csv',
               ova=constants.VALID_HTYPES + ['all'])
