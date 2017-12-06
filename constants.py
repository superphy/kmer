import os

#default filepaths to data
SALMONELLA = '/home/rboothman/Data/salmonella_amr/'
ECOLI = '/home/rboothman/Data/ecoli/'
OMNILOG_FASTA = '/home/rboothman/Data/ecomnilog/fasta/'
MORIA = '/home/rboothman/moria/enterobase_db/'
GENOME_REGIONS = '/home/rboothman/Data/genome_regions/binary_tables/'

#default genome_region table
GENOME_REGION_TABLE = '/home/rboothman/Data/genome_regions/binary_tables/binary_table.txt'

#omnilog data file
OMNILOG_DATA = '/home/rboothman/Data/ecomnilog/wide_format_header.txt'

#roary data file
ROARY = '/home/rboothman/Data/Roary/roary_results.csv'
#roary features used in paper
ROARY_VALID = './Data/PNAS_valid.txt'

#path to the source code
SOURCE = os.path.dirname(os.path.abspath(__file__)) + '/'

#path to default kmer count database
DB = SOURCE + 'database'

#default run config file
CONFIG = SOURCE + 'Data/config.yml'

#default file to store results to
OUTPUT = SOURCE + 'Data/run_results.yml'

#default filepaths to metadata sheets
SALMONELLA_METADATA = SOURCE + 'Data/amr_sorted.csv'
ECOLI_METADATA = SOURCE + 'Data/human_bovine.csv'
OMNILOG_METADATA= SOURCE + 'Data/omnilog_metadata.csv'
PREDICTIVE_RESULTS = SOURCE + 'Data/hb_train_predictiveresults.csv'
