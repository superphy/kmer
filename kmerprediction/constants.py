"""
Contains constants used in the other modules of the program. Defines things
such as default paths to data and metadata, this allows a user to update
everything from just one file ofter starting to use the program on their
computer.
"""

import numpy as np
import pandas as pd
import os

# default filepaths to data
SALMONELLA = '/home/rboothman/Data/salmonella_amr/'
ECOLI = '/home/rylan/Data/ecoli/fasta/'
OMNILOG_FASTA = '/home/rylan/Data/ecomnilog/fasta/'
MORIA = '/home/rboothman/moria/enterobase_db/'
GENOME_REGIONS = '/home/rboothman/Data/genome_regions/binary_tables/'

# default genome_region table
GENOME_REGION_TABLE = '/home/rylan/Data/binary_table.txt'

# omnilog data file
OMNILOG_DATA = '/home/rylan/Data/ecomnilog/wide_format_header.txt'

# roary data file
ROARY = '/home/rboothman/Data/Roary/roary_results.csv'
# roary features used in paper
ROARY_VALID = './Data/PNAS_valid.txt'

# path to the source code
SOURCE = os.path.dirname(os.path.abspath(__file__)) + '/'

# path to default kmer count database
DEFAULT_DB = SOURCE + 'database'

# default run config file
CONFIG = SOURCE + 'Data/config.yml'

# default file to store results to
OUTPUT = SOURCE + 'Data/run_results.yml'

# default filepaths to metadata sheets
SALMONELLA_METADATA = SOURCE + 'Data/amr_sorted.csv'
ECOLI_METADATA = SOURCE + 'Data/human_bovine.csv'
OMNILOG_METADATA = SOURCE + 'Data/omnilog_metadata.csv'
PREDICTIVE_RESULTS = SOURCE + 'Data/hb_train_predictiveresults.csv'
OMNILOG_WELLS = SOURCE + 'Data/omnilog_wells.csv'

MIN_FREQUENCY = 5
metadata = pd.read_csv(OMNILOG_METADATA)

sero, count = np.unique(metadata['Serotype'].values, return_counts=True)
VALID_SEROTYPES = [sero[x] for x, y in enumerate(count) if y >= MIN_FREQUENCY]

otype, count = np.unique(metadata['O type'].values, return_counts=True)
VALID_OTYPES = [otype[x] for x, y in enumerate(count) if y >= MIN_FREQUENCY]

htype, count = np.unique(metadata['H type'].values, return_counts=True)
VALID_HTYPES = [htype[x] for x, y in enumerate(count) if y >= MIN_FREQUENCY]

host, count = np.unique(metadata['Host'].values, return_counts=True)
VALID_HOSTS = [host[x] for x, y in enumerate(count) if y >= MIN_FREQUENCY]

DEFAULT_NAME = 'complete_results'
