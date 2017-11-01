from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from kmer_counter import count_kmers, get_counts
from utils import check_fasta, same_shuffle, shuffle
from utils import parse_genome_region_table, parse_metadata
from utils import get_human_path, get_bovine_path
import numpy as np
import pandas as pd
import csv
import os
import random

human_path = get_human_path()
bovine_path = get_bovine_path()


def get_kmer_us_uk_split(database, threeD, recount, k, l):
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
    x_train, y_train, x_test, y_test = parse_metadata(*sparams)

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


def get_kmer_us_uk_mixed(database, threeD, recount, k, l):
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


def get_genome_region_us_uk_mixed(table, threeD):
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


def get_genome_region_us_uk_split(table, threeD):
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
