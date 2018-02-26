"""
Recreates the omnilog analysis from the paper
"""
import yaml
import os
import sys
sys.path.append('../../')
from run import main
from kmer_counter import count_kmers
from generate_config import generate_config_files

def prep_db(path_to_db):
    path_to_fasta = constants.ECOLI
    all_files = [path_to_fasta + x for x in os.listdir(path_to_fasta)]
    count_kmers(7, 13, all_files, constants.DB, True)

def run_all():
    directory = './config_files/'
    results_dir = './results/'

    if not os.exists(directory):
        generate_config_files()

    if not os.exists(results_dir):
        os.makedirs(results_dir)

    all_files = [x for x in os.listdir(directory) if '.yml' in x]

    prep_db()

    for f in all_files:
        main(f, results_dir + '%s' % f, f.replace('.yml', ''))

if __name__ == '__main__':
    run_all()
