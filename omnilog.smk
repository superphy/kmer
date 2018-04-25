import os
from kmerprediction import constants

configfile: 'omnilog_config.yml'

ova_dict = {'Host': constants.VALID_HOSTS,
            'Htype': constants.VALID_HTYPES,
            'Otype': constants.VALID_OTYPES,
            'Serotype': constants.VALID_SEROTYPES,
            'Lineage': constants.VALID_LINEAGES,
            'Multiclass': 'all'}


def prediction_converter(prediction):
    if prediction == 'Multiclass':
        out = [x for x in config['prediction'] if x != 'Multiclass']
    else:
        out = prediction
    return out


def get_yaml(wc):
    yaml_dir = 'results/omnilog/yaml/'
    a = expand(yaml_dir + '{model}/omnilog/{selection}/{prediction}/{ova}/results.yml',
               model=config['model'],
               selection=config['selection'],
               prediction = prediction_converter(wc.prediction),
               ova=ova_dict[wc.prediction] + ['all'])
    b = expand(yaml_dir + '{model}/{k}mer_{filter}/{selection}/{prediction}/{ova}/results.yml',
               model=config['model'],
               k=config['k'],
               filter=config['filter'],
               selection=config['selection'],
               prediction = prediction_converter(wc.prediction),
               ova=ova_dict[wc.prediction] + ['all'])
    return a + b

def all_predictions():
    out = []
    for p in config['prediction']:
        if p != 'Multiclass':
            for o in ova_dict[p]:
                out.append('{}_{}'.format(p, o))
        else:
            for x in config['prediction']:
                if x != 'Multiclass':
                    out.append('{}_all'.format(x))
    return out


rule all:
    input:
        'manuscript/tables/complete_omnilog_results.md',
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
        get_yaml
    output:
        'manuscript/tables/omnilog_{prediction}_table.md'
    script:
        'scripts/omni_tables.py'


# Combine individual tables into one large table for the manuscript
rule complete_results_table:
    input:
        expand('manuscript/tables/omnilog_{p_class}_table.md',
               p_class=config['prediction'])
    output:
        'manuscript/tables/complete_omnilog_results.md'
    script:
        'scripts/omni_complete_table.py'

