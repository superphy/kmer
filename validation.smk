import os

configfile: 'config.yml'

rule all:
    input:
        'manuscript/tables/validation_results.md',
        'manuscript/images/validation_results.pdf',
        'manuscript/validation_results.tex'

# Generate the input config files for the Lupolova e. Coli analysis
rule config:
    output:
        expand('config_files/validation/{model}_{data}_{split}_{k}_{selection}.yml',
               model=config['model'],
               data=['kmer'],
               split=['split', 'mixed'],
               k=config['k_vals'],
               selection=config['selection'])
    script:
        'scripts/validation_config.py'

# Run kmerprediction.run.main on the specified input and ouput yaml
rule run:
    input:
        'config_files/{dir}/{analysis}.yml'
    output:
        'results/{dir}/yaml/{analysis}.yml'
    run:
        from kmerprediction.run import main
        main(input[0], output[0], input[0])

# Convert yaml output by run into pandas dataframes
rule data_frames:
    input:
        expand('results/validation/yaml/{model}_{data}_{split}_{k}_{selection}.yml',
               model=config['model'],
               data=['kmer'],
               split=['split', 'mixed'],
               k=config['k_vals'],
               selection=config['selection'])
    output:
        'results/validation/DataFrames/results.csv'
    script:
        'scripts/validation_data_frames.py'

# Make figures for manuscript
rule figures:
    input:
        'results/validation/DataFrames/results.csv'
    output:
        'manuscript/images/validation_results.pdf'
    script:
        'scripts/validation_figures.py'

# Make markdown tables for manuscript
rule tables:
    input:
        expand('results/validation/yaml/{model}_{data}_{split}_{k}_{selection}.yml',
               model=config['model'],
               data=['kmer'],
               split=['split', 'mixed'],
               k=config['k_vals'],
               selection=config['selection'])
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
        'validation_results.py'
