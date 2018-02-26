rule generate_us_uk_config:
    output:
        expand('config_files/{model}_{data}_{split}.yml',
               model=['neural_network', 'support_vector_machine', 'random_forest'],
               data=['kmer', 'genome'],
               split=['split', 'mixed'])
    script:
        'snakemake_scripts/generate_us_uk_config.py'

rule generate_omni_config:
    output:
        expand('config_files/{model}_Host_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'Bovine', 'Human', 'Water']),
        expand('config_files/{model}_Serotype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'O103:H2', 'O113:H21', 'O121:H19', 'O145:NM',
                           'O157:H7', 'O26:H11', 'O45:H2', 'O91:H21']),
        expand('config_files/{model}_Otype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'O103', 'O104', 'O113', 'O121', 'O145',
                           'O157', 'O26', 'O45', 'O91']),
        expand('config_files/{model}_Htype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'H11', 'H19', 'H21', 'H2', 'H7', 'NM'])
    script:
        'snakemake_scripts/generate_omni_config.py'

rule run_single:
    input:
        'config_files/{analysis}.yml'
    output:
        'results/{analysis}.yml'
    run:
        import os
        from kmerprediction.run import main
        from kmerprediction import constants
        from kmerprediction.kmer_counter import get_counts, count_kmers
        if 'split' in input[0] or 'mixed' in input[0]:
            fasta_path = constants.ECOLI
        else:
            fasta_path = constants.OMNILOG_FASTA
        fasta_files = [fasta_path + x for x in os.listdir(fasta_path)]
        try:
            a = get_counts(fasta_files, constants.DB)
        except:
            count_kmers(7, 13, fasta_files, constants.DB, True)
        name = input[0].replace('config_files/', '')
        main(input[0], output[0], name)

rule run_us_uk_analysis:
    input:
        expand('results/{model}_{data}_{split}.yml',
               model=['neural_network', 'support_vector_machine', 'random_forest'],
               data=['kmer', 'genome'],
               split=['split', 'mixed'])

rule run_omni_analysis:
    input:
        expand('results/{model}_Host_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'Bovine', 'Human', 'Water']),
        expand('results/{model}_Serotype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'O103:H2', 'O113:H21', 'O121:H19', 'O145:NM',
                           'O157:H7', 'O26:H11', 'O45:H2', 'O91:H21']),
        expand('results/{model}_Otype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'O103', 'O104', 'O113', 'O121', 'O145',
                           'O157', 'O26', 'O45', 'O91']),
        expand('results/{model}_Htype_{data}_{prediction}.yml',
               model=['support_vector_machine', 'neural_network', 'random_forest'],
               data=['kmer', 'omni'],
               prediction=['all', 'H11', 'H19', 'H21', 'H2', 'H2', 'NM'])
