import os
import pandas as pd
from kmerprediction import constants
from kmerprediction.run import main
from kmerprediction.complete_kmer_counter import count_kmers, get_counts
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

try:
    a = get_counts(fasta_files, output_db)
except Exception as e:
    print(e)
    print('RunSingleWarning: get_counts failed, attempting recount')
    count_kmers(k, fasta_files, complete_db, output_db)

main(snakemake.input[0], snakemake.output[0], snakemake.input[0])
