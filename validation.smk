import os

configfile: 'config.yml'

def get_yaml():
    out = expand('results/validation/yaml/{model}/7mer_filtered_13/' +
                 '{dataset}/{selection}/results.yml',
                 model=config['model'], dataset=config['dataset'],
                 selection=config['selection'])
    out += expand('results/validation/yaml/{model}/{k}mer_{filter}/' +
                 '{dataset}/{selection}/results.yml',
                 model=config['model'], k=config['k'],
                 dataset=config['dataset'], selection=config['selection'],
                 filter=config['filter'])
    return out


rule all:
    input:
        'manuscript/tables/validation_results.md',
        'manuscript/images/validation_results.pdf',
        'manuscript/validation_results.tex',
        'results/validation/features/7mer_filtered_13_features.csv',
        expand('results/validation/features/{k}mer_{filter}_features.csv',
               k=config['k'], filter=config['filter']),
        expand('results/validation/histogram/{k}mer_histogram.csv', k=config['k']),
        expand('results/validation/kmer_appearances/{k}mer_appearances.csv', k=config['k'])


# Generate the input config files for the Lupolova e. Coli analysis
rule config:
    output:
        'config_files/validation/{model}/{k}mer_{filter}/{dataset}/{selection}/config.yml'
    threads: 24
    script:
        'scripts/validation_config.py'


# Run kmerprediction.run.main on the specified input and ouput yaml
rule run:
    input:
        'config_files/validation/{model}/{data}/{dataset}/{selection}/config.yml'
    output:
        'results/validation/yaml/{model}/{data}/{dataset}/{selection}/results.yml'
    threads: 2
    run:
        from kmerprediction.run import main
        main(input[0], output[0], input[0])


# Convert yaml output by run into pandas dataframes
rule data_frames:
    input:
        get_yaml()
    output:
        'results/validation/DataFrames/results.csv'
    script:
        'scripts/validation_dfs.py'


# Make figures for manuscript
rule figures:
    input:
        'results/validation/DataFrames/results.csv'
    output:
        'manuscript/images/validation_results.pdf'
    script:
        'scripts/validation_figs.py'


# Make markdown tables for manuscript
rule tables:
    input:
        get_yaml()
    output:
        'manuscript/tables/validation_results.md'
    script:
        'scripts/validation_tables.py'


# Make LaTeX macros to expand validation results in manuscript
rule macros:
    input:
        'results/validation/DataFrames/results.csv'
    output:
        'manuscript/validation_results.tex'
    script:
        'scripts/validation_results.py'


# Make k-mer frequency distribution
rule histograms:
    input:
        '/home/rylan/Data/lupolova_data/complete_database/complete_{k}-mer_DB'
    output:
        'results/validation/histogram/{k}mer_histogram.csv'
    script:
        'scripts/kmer_histogram.py'


# compare importance of features for different models/datasets
rule features:
    input:
        expand('results/validation/yaml/{model}/{{k}}mer_{{filter}}/' +
               '{dataset}/{selection}/results.yml',
               model=config['model'], dataset=config['dataset'],
               selection=config['selection']),
    output:
        'results/validation/features/{k}mer_{filter}_features.csv'
    script:
        'scripts/validation_features.py'

rule filtered_features:
    input:
        expand('results/validation/yaml/{model}/7mer_filtered_13/' +
               '{dataset}/{selection}/results.yml',
               model=config['model'], dataset=config['dataset'],
               selection=config['selection'])
    output:
        'results/validation/features/7mer_filtered_13_features.csv'
    script:
        'scripts/validation_features.py'

# Create histograms for file counts of different kmer lengths
rule kmer_appearance:
    input:
        '/home/rylan/Data/lupolova_data/complete_database/complete_{k}-mer_DB'
    output:
        'results/validation/kmer_appearances/{k}mer_appearances.csv'
    script:
        'scripts/kmer_appearances.py'
