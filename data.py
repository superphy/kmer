from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from kmer_counter import count_kmers, get_counts
from utils import check_fasta, same_shuffle, shuffle
from utils import parse_genome_region_table, parse_metadata
from utils import get_human_path, get_bovine_path, parse_json
import numpy as np
import pandas as pd
import csv
import os
import random

human_path = get_human_path()
bovine_path = get_bovine_path()


def get_kmer_us_uk_split(database="database", threeD=True, recount=False, k=7, l=13):
    """
    Parameters:
        database:   lmdb database to store kmer counts.
        threeD:     If True the data is made three dimensional.
        recount:    If True the kmers are recounted.
        k:          Size of kmer to be counted. Ignored if recount is false.
        l:          kmer cutoff value. Ignored if recount is false.
    Returns:
        x_train, y_train, x_test, y_test

        kmer data ready to be input into a ml model and recreate the Lupolova
        et. al paper.
    """
    params = ['human_bovine.csv', 'Human', 'Bovine', human_path, bovine_path,
              'Train', 'Test', '.fasta']
    x_train, y_train, x_test, y_test = parse_metadata(*params)

    if recount:
        allfiles = x_train + x_test
        count_kmers(k, l, allfiles, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(list(x_test), dtype='float64')

    scaler = MinMaxScaler(feature_range=(-1,1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if threeD:
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def get_kmer_us_uk_mixed(database="database", threeD=True, recount=False, k=7, l=13):
    """
    Parameters:
        database:   lmdb database to store kmer counts.
        threeD:     If True the data is made three dimensional.
        recount:    If True the kmers are recounted.
        k:          Size of kmer to be counted. Ignored if recount is false.
        l:          kmer cutoff value. Ignored if recount is false.
    Returns:
        x_train, y_train, x_test, y_test

        kmer data ready to be input into a ml model, with us/uk data shuffled
        together.
    """
    params = ['human_bovine.csv', 'Human', 'Bovine', human_path, bovine_path,
              '', '', '.fasta']
    y_train, y_train, x_test, y_test = parse_metadata(*params)

    if recount:
        count_kmers(k, l, genomes, database)

    x_train = get_counts(x_train, database)
    x_test = get_counts(x_test, database)

    scaler = MinMaxScaler(feature_range=(-1,1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if threeD:
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def get_genome_region_us_uk_mixed(table='binary_table.txt', threeD=True):
    """
    Parameters:
        table:   binary_table.txt output from panseq.
        threeD:  If True the data is made three dimensional.
    Returns:
        x_train, y_train, x_test, y_test

        binary genome region presence absence data ready to be input into a ml
        model, with us/uk data shuffled together.
    """
    params = ['human_bovine.csv', 'Human', 'Bovine']
    x_train, y_train, x_test, y_test = parse_genome_region_table(table, *params)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    if threeD:
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def get_genome_region_us_uk_split(table='binary_table.txt', threeD=True):
    """
    Parameters:
        table:   binary_table.txt output from panseq.
        threeD:  If True the data is made three dimensional.
    Returns:
        x_train, y_train, x_test, y_test

        binary genome region presence absence data ready to be input into a ml
        model to recreate the Lupoloval et. al paper.
    """
    params = ['human_bovine.csv', 'Human', 'Bovine', '', '', 'Train', 'Test']
    x_train, y_train, x_test, y_test = parse_genome_region_table(table, *params)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    if threeD:
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def get_kmer_data_from_json(database='database',threeD=True,recount=False,k=7,
                            l=13,path='/home/rboothman/moria/entero_db/',
                            suffix='.fasta',key='assembly_barcode',*json):
    """
    Parameters:
        database,threeD,recount,k,l: See get_kmer_us_uk_split
        path: Prefix to be added to the beginning of every fastafile in the json
        suffix: Suffix to be added to the end of every fasta file in the json
        *json: Two or four json files, If two the first should contain only
               positive genomes and the second should contain only negative
               genomes. If four the first json file should contain all positive
               training genomes, the second all negative training genomes, the
               third all positive test genomes, the fourth all negative test
               genomes.
    Returns:
        x_train, y_train, x_test, y_tes ready to be input into a ml model.

    """
    files = parse_json(path, suffix, key, *json)

    if len(files) == 4:
        x_train, y_train = shuffle(files[0], files[1], 1, 0)
        x_test, y_test = shuffle(files[2], files[3], 1, 0)
    elif len(files) == 2:
        x_train, y_train = shuffle(files[0], files[1], 1, 0)
        cutoff = int(0.8*len(x_train))
        x_test = x_train[cutoff:]
        y_test = y_train[cutoff:]
        x_train = x_train[:cutoff]
        y_train = y_train[:cutoff]

    if recount:
        all_files = x_train + x_test
        count_kmers(k, l, all_files, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(list(x_test), dtype='float64')

    scaler = MinMaxScaler(feature_range=(-1,1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if threeD:
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test
