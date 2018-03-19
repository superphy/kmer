import os
import pandas as pd
from kmerprediction import constants
from kmerprediction.run import main
from kmerprediction.kmer_counter4 import count_kmers, get_counts
from kmerprediction.utils import parse_metadata

if snakemake.wildcards.dir == 'US_UK':
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'train_header': None,
              'validate': True}
elif snakemake.wildcards.dir == 'omni':
    kwargs = {'extra_header': 'WGS',
              'extra_label': 1,
              'fasta_header': 'Strain',
              'label_header': 'Host',
              'metadata': constants.OMNILOG_METADATA,
              'prefix': constants.OMNILOG_FASTA,
              'suffix': '.fasta',
              'train_header': None}

(x_train, y_train, x_test, y_test) = parse_metadata(**kwargs)
fasta_files = x_train + x_test

name = snakemake.input[0].split('/')[-1].split('_')
k = int(name[-2])
N = int(name[-1].replace('.yml', ''))

try:
    a = get_counts(k, N, fasta_files, constants.OUTPUT_DIR)
except Exception as e:
    count_kmers(k, N, fasta_files, constants.OUTPUT_DIR, True)

main(snakemake.input[0], snakemake.output[0], snakemake.input[0])
