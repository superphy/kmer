import os
from kmerprediction import constants

configfile: 'omnilog_config.yml'

ova_dict = {'Host': constants.VALID_HOSTS + ['all'],
            'Htype': constants.VALID_HTYPES + ['all'],
            'Otype': constants.VALID_OTYPES + ['all'],
            'Serotype': constants.VALID_SEROTYPES + ['all'],
            'Lineage': constants.VALID_LINEAGES + ['all'],
            'Multiclass': 'all'}


def get_yaml(wc):
    yaml_dir = 'results/omnilog/yaml/'
    a = expand(yaml_dir + '{model}/omnilog/{selection}/{prediction}/{ova}/results.yml',
               model=config['model'],
               selection=config['selection'],
               prediction = wc.prediction,
               ova=ova_dict[wc.prediction])
    b = expand(yaml_dir + '{model}/{k}mer_{filter}/{selection}/{prediction}/{ova}/results.yml',
               model=config['model'],
               k=config['k'],
               filter=config['filter'],
               selection=config['selection'],
               prediction = wc.prediction,
               ova=ova_dict[wc.prediction])
    return a + b

def all_predictions():
    out = []
    for p in config['prediction']:
            for o in ova_dict[p]:
                out.append('{}/{}'.format(p, o))
    return out


rule all:
    input:
        'manuscript/tables/complete_omnilog_results.md',
        'manuscript/omnilog_results.tex',
        expand('manuscript/images/omnilog_{figure}.pdf',
               figure=config['prediction']),
        expand('manuscript/images/important_features/{prediction}.pdf',
                prediction=all_predictions())


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


# Convert the Omnilog results into pandas DataFrames to plot the model accuracies
rule dataframes:
    input:
        get_yaml
    output:
        'results/omnilog/DataFrames/{prediction}.csv'
    script:
        'scripts/omni_dataframes.py'


# Make figures for the manuscript of the binary results.
rule figures:
    input:
        'results/omnilog/DataFrames/{prediction}.csv'
    output:
        'manuscript/images/omnilog_{prediction}.pdf'
    script:
        'scripts/omni_figures.py'


# Convert the important feature data returned by the SVM and RF models into pandas DataFrames
rule important_features_data_frames:
    input:
        expand('results/omnilog/yaml/{model}/omnilog/{selection}/{{prediction}}/{{ova}}/results.yml',
               model=config['important_features_models'],
               selection=config['selection'])
    output:
        'results/omnilog/important_features/{prediction}/{ova}.csv'
    script:
        'scripts/omni_features.py'


# Make figures for the manuscript of the important feature results
rule important_features_figures:
    input:
        'results/omnilog/important_features/{prediction}/{ova}.csv'
    output:
        'manuscript/images/important_features/{prediction}/{ova}.pdf'
    script:
        'scripts/omni_features_figs.py'


# Convert dataframes to markdown tables for the manuscript
rule tables:
    input:
        get_yaml
    output:
        'manuscript/tables/omnilog_{prediction}_table.md'
    script:
        'scripts/omni_tables.py'


# Combine individual tables into one large table for the manuscript
rule complete_results_table:
    input:
        expand('manuscript/tables/omnilog_{prediction}_table.md',
               prediction=config['prediction'])
    output:
        'manuscript/tables/complete_omnilog_results.md'
    script:
        'scripts/omni_complete_table.py'


# Make value to insert into manuscript
rule macros:
    input:
        expand('results/omnilog/DataFrames/{prediction}.csv',
               prediction=config['prediction'])
    output:
        'manuscript/omnilog_results.tex'
    script:
        'scripts/omnil_results.py'



