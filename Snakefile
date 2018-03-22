import os
from kmerprediction import constants

ova_dict = {'Host': constants.VALID_HOSTS,
            'Htype': constants.VALID_HTYPES,
            'Otype': constants.VALID_OTYPES,
            'Serotype': constants.VALID_SEROTYPES}

rule generate_us_uk_config: # Generate the input config files for the US/UK e. Coli analysis
    output:
        expand('config_files/US_UK/{model}_{data}_{split}.yml',
               model=['neural_network', 'support_vector_machine', 'random_forest'],
               data=['kmer', 'genome'],
               split=['split', 'mixed'])
    script:
        'scripts/us_uk_config.py'


rule generate_omni_config: # Generate the input config files for the Omnilog analysis
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


rule run_single: # Run kmerprediction.run.main on the specified input and ouput yml
    input:
        'config_files/{dir}/{analysis}.yml'
    output:
        'results/{dir}/yaml/{analysis}.yml'
    script:
        'scripts/run_single.py'


# Convert the Omnilog binary results into pandas DataFrames to plot the model accuracies
rule make_omni_binary_data_frames:
    input:
        lambda wc: expand('results/omni/yaml/{model}_{prediction}_{data}_{ova}.yml',
                                  model=['support_vector_machine', 'neural_network',
                                         'random_forest'],
                                  data=['kmer', 'omni'],
                                  ova=ova_dict[wc.prediction],
                                  prediction=wc.prediction)
    output:
        'results/omni/DataFrames/{prediction}.csv'
    script:
        'scripts/make_binary_data_frames.py'


# Convert the Omnilog multiclass results into a pandas DataFrame to plot the model accuracies
rule make_omni_multiclass_data_frames:
    input:
        expand('results/omni/yaml/{model}_{prediction}_{datatype}_all.yml',
               model=['random_forest', 'support_vector_machine', 'neural_network'],
               datatype=['kmer', 'omni'],
               prediction=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'results/omni/DataFrames/Multiclass.csv'
    script:
        'scripts/make_multiclass_data_frames.py'


rule plot_binary_omni:
    input:
        'results/omni/DataFrames/{prediction}.csv'
    output:
        'results/omni/Figures/{prediction, (Host)|(Otype)|(Htype)|(Serotype)}.pdf'
    script:
        'scripts/plot_binary.py'


rule plot_multiclass_omni:
    input:
        'results/omni/DataFrames/Multiclass.csv'
    output:
        'results/omni/Figures/Multiclass.pdf'
    script:
        'scripts/plot_multiclass.py'


rule plot_all_omnilog:
    input:
        expand('results/omni/Figures/{figure}.pdf',
               figure=['Multiclass', 'Host', 'Htype', 'Otype', 'Serotype'])


# Convert the important feature data returned by the SVM and RF models into pandas DataFrames
rule important_features_data_frame:
    input:
        expand('results/omni/yaml/{model}_{{prediction}}_omni_{{ova}}.yml',
               model=['random_forest', 'support_vector_machine'])
    output:
        'results/omni/Important_Features/{prediction}_{ova}.csv'
    script:
        'scripts/important_features.py'


rule plot_important_features:
    input:
        'results/omni/Important_Features/{prediction}_{ova}.csv'
    output:
        'results/omni/Figures/Important_Features/{prediction}_{ova}.pdf'
    script:
        'scripts/plot_important_features.py'


rule important_features_analysis:
    input:
        expand('results/omni/Figures/Important_Features/Host_{ova}.pdf',
               ova=constants.VALID_HOSTS + ['all']),
        expand('results/omni/Figures/Important_Features/Serotype_{ova}.pdf',
               ova=constants.VALID_SEROTYPES + ['all']),
        expand('results/omni/Figures/Important_Features/Otype_{ova}.pdf',
               ova=constants.VALID_OTYPES + ['all']),
        expand('results/omni/Figures/Important_Features/Htype_{ova}.pdf',
               ova=constants.VALID_HTYPES + ['all'])

rule make_us_uk_data_frame:
    input:
        expand('results/US_UK/yaml/{model}_{data}_{split}.yml',
               model=['neural_network', 'support_vector_machine', 'random_forest'],
               data=['kmer', 'genome'],
               split=['split', 'mixed'])
    output:
        'results/US_UK/DataFrames/results.csv'
    script:
        'scripts/make_us_uk_data_frame.py'

rule plot_us_uk:
    input:
        'results/US_UK/DataFrames/results.csv'
    output:
        'results/US_UK/Figures/results.pdf'
    script:
        'scripts/plot_us_uk.py'

rule make_omni_table:
    input:
        lambda wc : expand('results/omni/yaml/{model}_{p_class}_{datatype}_{p}.yml',
                           model=['random_forest', 'support_vector_machine',
                                  'neural_network'],
                           datatype=['kmer', 'omni'],
                           p_class = wc.p_class,
                           p=ova_dict[wc.p_class] + ['all'])
    output:
        'results/omni/Tables/{p_class}_table.md'
    script:
        'scripts/make_omni_tables.py'

rule make_all_omni_tables:
    input:
        expand('results/omni/Tables/{p_class}_table.md',
               p_class=['Host', 'Htype', 'Otype', 'Serotype'])
    output:
        'results/omni/Tables/complete_results.md'
    run:
        import pandas as pd
        frames = [pd.read_csv(x, sep='|') for x in input]
        complete = pd.concat(frames, ignore_index=True)
        complete = complete[complete.Model != '---']
        new = pd.DataFrame(columns=complete.columns)
        new.loc[0] = ['---',]*len(complete.columns)
        output_df = pd.concat([new, complete], ignore_index=True)
        output_df.to_csv(output[0], sep='|', index=False)

rule make_us_uk_table:
    input:
        expand('results/US_UK/yaml/{model}_{data}_{split}.yml',
               model=['neural_network', 'support_vector_machine', 'random_forest'],
               data=['kmer', 'genome'],
               split=['split', 'mixed'])
    output:
        'results/US_UK/Tables/complete_results.md'
    script:
        'scripts/make_us_uk_table.py'

rule omnilog_analysis:
    input:
        'results/omni/Tables/complete_results.md',
        expand('results/omni/Figures/{figure}.pdf',
               figure=['Multiclass', 'Host', 'Htype', 'Otype', 'Serotype'])

rule us_uk_analysis:
    input:
        'results/US_UK/Tables/complete_results.md',
        'results/US_UK/Figures/results.pdf'

rule values_for_paper:
    input:
        expand('results/omni/DataFrames/{df}.csv',
               df=['Host', 'Htype', 'Multiclass','Otype', 'Serotype']),
        'results/US_UK/DataFrames/results.csv'
    output:
        'results/tex_files/values.pkl'
    script:
        'scripts/values_for_paper.py'

rule generate_tex:
    input:
        'results/tex_files/values.pkl'
    output:
        'results/tex_files/values.tex'
    script:
        'scripts/make_macros.py'
