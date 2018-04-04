import os
from kmerprediction import constants

configfile: 'config.yml'

ova_dict = {'Host': constants.VALID_HOSTS,
            'Htype': constants.VALID_HTYPES,
            'Otype': constants.VALID_OTYPES,
            'Serotype': constants.VALID_SEROTYPES}

rule all:
    input:
        'manuscript/tables/complete_omnilog_results.md',
        expand('manuscript/images/{figure}.pdf',
               figure=['Multiclass', 'Host', 'Htype', 'Otype', 'Serotype'])
        expand('manuscript/images/important_features/Host_{ova}.pdf',
               ova=constants.VALID_HOSTS + ['all']),
        expand('manuscript/images/important_features/Serotype_{ova}.pdf',
               ova=constants.VALID_SEROTYPES + ['all']),
        expand('manuscript/images/important_features/Otype_{ova}.pdf',
               ova=constants.VALID_OTYPES + ['all']),
        expand('manuscript/images/important_features/Htype_{ova}.pdf',
               ova=constants.VALID_HTYPES + ['all'])


# Generate the input config files for the Omnilog analysis
rule config:
    output:
        expand('config_files/omni/{model}_Host_{data}_{prediction}.yml',
               model=config['model'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_HOSTS + ['all']),
        expand('config_files/omni/{model}_Serotype_{data}_{prediction}.yml',
               model=config['model'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_SEROTYPES + ['all']),
        expand('config_files/omni/{model}_Otype_{data}_{prediction}.yml',
               model=config['model'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_OTYPES + ['all']),
        expand('config_files/omni/{model}_Htype_{data}_{prediction}.yml',
               model=config['model'],
               data=['kmer', 'omni'],
               prediction=constants.VALID_HTYPES + ['all'])
    script:
        'scripts/omni_config.py'


# Run kmerprediction.run.main on the specified input and ouput yaml
rule run:
    input:
        'config_files/{dir}/{analysis}.yml'
    output:
        'results/{dir}/yaml/{analysis}.yml'
    run:
        from kmerprediction.run import main
        main(input[0], output[0], input[0])


# Convert the Omnilog binary results into pandas DataFrames to plot the model accuracies
rule binary_data_frames:
    input:
        lambda wc: expand('results/omni/yaml/{model}_{prediction}_{data}_{ova}.yml',
                          model=config['model'],
                          data=['kmer', 'omni'],
                          ova=ova_dict[wc.prediction],
                          prediction=wc.prediction)
    output:
        'results/omni/DataFrames/{prediction}.csv'
    script:
        'scripts/omni_binary_dfs.py'


# Convert the Omnilog multiclass results into a pandas DataFrame to plot the model accuracies
rule multiclass_data_frames:
    input:
        expand('results/omni/yaml/{model}_{prediction}_{datatype}_all.yml',
               model=config['model'],
               datatype=['kmer', 'omni'],
               prediction=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'results/omni/DataFrames/Multiclass.csv'
    script:
        'scripts/omni_multi_dfs.py'


# Make figures for the manuscript of the binary results.
rule binary_figures:
    input:
        'results/omni/DataFrames/{prediction}.csv'
    output:
        'manuscript/images/omnilog_{prediction, (Host)|(Otype)|(Htype)|(Serotype)}.pdf'
    script:
        'scripts/omni_binary_figs.py'


# Make figures for the manuscript of the multiclass results
rule multiclass_figures:
    input:
        'results/omni/DataFrames/Multiclass.csv'
    output:
        'manuscript/images/omnilog_Multiclass.pdf'
    script:
        'scripts/omni_multi_figs.py'


# Convert the important feature data returned by the SVM and RF models into pandas DataFrames
rule important_features_data_frames:
    input:
        expand('results/omni/yaml/{model}_{{prediction}}_omni_{{ova}}.yml',
               model=config['important_features_model'])
    output:
        'results/omni/important_features/{prediction}_{ova}.csv'
    script:
        'scripts/important_features.py'


# Make figures for the manuscript of the important feature results
rule important_features_figures:
    input:
        'results/omni/important_features/{prediction}_{ova}.csv'
    output:
        'manuscript/images/important_features/{prediction}_{ova}.pdf'
    script:
        'scripts/important_features_figs.py'


# Convert dataframes to markdown tables for the manuscript
rule tables:
    input:
        lambda wc : expand('results/omni/yaml/{model}_{p_class}_{datatype}_{p}.yml',
                           model=config['model'],
                           datatype=['kmer', 'omni'],
                           p_class = wc.p_class,
                           p=ova_dict[wc.p_class] + ['all'])
    output:
        'manuscript/tables/omnilog_{p_class}_table.md'
    script:
        'scripts/omni_tables.py'


# Combine individual tables into one large table for the manuscript
rule complete_results_table:
    input:
        expand('results/omni/Tables/{p_class}_table.md',
               p_class=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'manuscript/tables/complete_omnilog_results.md'
    script:
        'scripts/omni_complete_table.py'

