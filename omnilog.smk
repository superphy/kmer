import os
from kmerprediction import constants

configfile: 'omnilog_config.yml'

ova_dict = {'Host': constants.VALID_HOSTS,
            'Htype': constants.VALID_HTYPES,
            'Otype': constants.VALID_OTYPES,
            'Serotype': constants.VALID_SEROTYPES}

rule all:
    input:
        'manuscript/tables/complete_omnilog_results.md',
        expand('manuscript/images/omnilog_{figure}.pdf',
               figure=['Multiclass', 'Host', 'Htype', 'Otype', 'Serotype']),
        expand('manuscript/images/important_features/Host_{ova}.pdf',
               ova=constants.VALID_HOSTS + ['all']),
        expand('manuscript/images/important_features/Serotype_{ova}.pdf',
               ova=constants.VALID_SEROTYPES + ['all']),
        expand('manuscript/images/important_features/Otype_{ova}.pdf',
               ova=constants.VALID_OTYPES + ['all']),
        expand('manuscript/images/important_features/Htype_{ova}.pdf',
               ova=constants.VALID_HTYPES + ['all'])


# Generate the input config files for the Omnilog analysis
rule kmer_config:
    output:
        'config_files/omnilog/{model}/{k}mer_{filter}/{selection}/{prediction}/{ova}/config.yml'
    script:
        'scripts/omni_config.py'

rule omnilog_config:
    output:
        'config_files/omnilog/{model}/omnilog/{selection}/{prediction}/{ova}/config.yml'
    script:
        'scripts/omni_config.py'


# Run kmerprediction.run.main on the specified input and ouput yaml
rule run:
    input:
        'config_files/omnilog/{model}/{data}/{selection}/{prediction}/{ova}/config.yml'
    output:
        'results/omnilog/yaml/{model}/{data}/{selection}/{prediction}/{ova}/results.yml'
    run:
        from kmerprediction.run import main
        main(input[0], output[0], input[0])


# Convert the Omnilog binary results into pandas DataFrames to plot the model accuracies
rule binary_data_frames:
    input:
        lambda wc: expand('results/omnilog/yaml/{model}/{k}mer_{filter}/{selection}/' +
                          '{prediction}/{ova}/results.yml',
                          model=config['model'],
                          k=config['k'],
                          filter=config['filter'],
                          selection=config['selection'],
                          prediction=wc.prediction,
                          ova=ova_dict[wc.prediction]),
        lambda wc: expand('results/omnilog/yaml/{model}/omnilog/{selection}/' +
                          '{prediction}/{ova}/results.yml',
                          model=config['model'],
                          selection=config['selection'],
                          prediction=wc.prediction,
                          ova=ova_dict[wc.prediction])
    output:
        'results/omnilog/DataFrames/{prediction}.csv'
    script:
        'scripts/omni_binary_dfs.py'


# Convert the Omnilog multiclass results into a pandas DataFrame to plot the model accuracies
rule multiclass_data_frames:
    input:
        expand('results/omnilog/yaml/{model}/{k}mer_{filter}/{selection}/' +
               '{prediction}/all/results.yml',
               model=config['model'],
               k=config['k'],
               filter=config['filter'],
               selection=config['selection'],
               prediction=['Host', 'Htype', 'Otype', 'Serotype']),
        expand('results/omnilog/yaml/{model}/omnilog/{selection}/' +
               '{prediction}/all/results.yml',
               model=config['model'],
               selection=config['selection'],
               prediction=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'results/omnilog/DataFrames/Multiclass.csv'
    script:
        'scripts/omni_multi_dfs.py'


# Make figures for the manuscript of the binary results.
rule binary_figures:
    input:
        'results/omnilog/DataFrames/{prediction}.csv'
    output:
        'manuscript/images/omnilog_{prediction, (Host)|(Otype)|(Htype)|(Serotype)}.pdf'
    script:
        'scripts/omni_binary_figs.py'


# Make figures for the manuscript of the multiclass results
rule multiclass_figures:
    input:
        'results/omnilog/DataFrames/Multiclass.csv'
    output:
        'manuscript/images/omnilog_Multiclass.pdf'
    script:
        'scripts/omni_multi_figs.py'


# Convert the important feature data returned by the SVM and RF models into pandas DataFrames
rule important_features_data_frames:
    input:
        expand('results/omnilog/yaml/{model}/omnilog/{selection}/{{prediction}}/{{ova}}/results.yml',
               model=config['important_features_models'],
               selection=config['selection'])
    output:
        'results/omnilog/important_features/{prediction}_{ova}.csv'
    script:
        'scripts/omni_features.py'


# Make figures for the manuscript of the important feature results
rule important_features_figures:
    input:
        'results/omnilog/important_features/{prediction}_{ova}.csv'
    output:
        'manuscript/images/important_features/{prediction}_{ova}.pdf'
    script:
        'scripts/omni_features_figs.py'


# Convert dataframes to markdown tables for the manuscript
rule tables:
    input:
        lambda wc : expand('results/omnilog/yaml/{model}/omnilog/{selection}/' +
                           '{prediction}/{ova}/results.yml',
                           model=config['model'],
                           selection=config['selection'],
                           prediction = wc.prediction,
                           ova=ova_dict[wc.prediction] + ['all']),
        lambda wc : expand('results/omnilog/yaml/{model}/{k}mer_{filter}/' +
                           '{selection}/{prediction}/{ova}/results.yml',
                           model=config['model'],
                           k=config['k'],
                           filter=config['filter'],
                           selection=config['selection'],
                           prediction = wc.prediction,
                           ova=ova_dict[wc.prediction] + ['all'])
    output:
        'manuscript/tables/omnilog_{prediction}_table.md'
    script:
        'scripts/omni_tables.py'


# Combine individual tables into one large table for the manuscript
rule complete_results_table:
    input:
        expand('manuscript/tables/omnilog_{p_class}_table.md',
               p_class=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'manuscript/tables/complete_omnilog_results.md'
    script:
        'scripts/omni_complete_table.py'

