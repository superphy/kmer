import os
import pandas as pd
from kmerprediction import constants
from kmerprediction.run import main
from kmerprediction.kmer_counter import count_kmers, get_counts

if snakemake.wildcards.dir == 'US_UK':
    fasta_path = constants.ECOLI
elif snakemake.wildcards.dir == 'omni':
    fasta_path = constants.OMNILOG_FASTA

fasta_files = [fasta_path + x for x in os.listdir(fasta_path)]

try:
    a = get_counts(fasta_files, constants.DB)
except:
    count_kmers(7, 13, fasta_files, constants.DB, True)

main(snakemake.input[0], snakemake.output[0], snakemake.input[0])
