from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from kmer_counter import count_kmers, get_counts
from utils import check_fasta, same_shuffle, shuffle, flatten, make3D, setup_files
from utils import parse_metadata, parse_json
import numpy as np
import pandas as pd
import csv
import os
import random


filepath = '/home/rboothman/Data/ecoli/'


def get_methods():
    output = {'kmer_split': get_kmer_us_uk_split,
              'kmer_mixed': get_kmer_us_uk_mixed,
              'genome_mixed': get_genome_region_us_uk_mixed,
              'genome_split': get_genome_region_us_uk_split,
              'kmer_json': get_kmer_from_json,
              'kmer_directory': get_kmer_from_directory,
              'genome_filtered': get_genome_pre_filtered,
              'genome_custom_filtered': get_genome_custom_filtered,
              'genome_custom': get_genome_regions,
              'salmonella': get_salmonella_kmer,
              'kmer_custom': get_kmer}
    return output


def get_kmer(args, database="database", recount=False, k=7, l=13):
    """
    Parameters:
        args:       The arguments to pass to parse_metadata.
        database:   lmdb database to store kmer counts.
        threeD:     If True the data is made three dimensional.
        recount:    If True the kmers are recounted.
        k:          Size of kmer to be counted. Ignored if recount is false.
        l:          kmer cutoff value. Ignored if recount is false.
    Returns:
        x_train, y_train, x_test, y_test
    """
    x_train, y_train, x_test, y_test = parse_metadata(*args)

    if recount:
        allfiles = x_train + x_test
        count_kmers(k, l, allfiles, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(list(x_test), dtype='float64')

    return (x_train, y_train, x_test, y_test)


def get_genome(args, table='Data/binary_table.txt', sep=None):
    """
    Parameters:
        args:    The arguments to pass to parse_metadata.
        table:   binary_table.txt output from panseq.
    Returns:
        x_train, y_train, x_test, y_test
    """
    labels = parse_metadata(*args)
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    x_train = []
    x_test = []
    if sep == None:
        data = pd.read_csv(table, sep=sep, engine='python', index_col=0)
    else:
        data = pd.read_csv(table, sep=sep, index_col=0)

    for header in x_train_labels:
        x_train.append(data[header].tolist())

    for header in x_test_labels:
        x_test.append(data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return (x_train, y_train, x_test, y_test)


def get_kmer_us_uk_split(database="database", recount=False, k=7, l=13):
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
    kwargs = {'prefix': filepath, 'suffix': '.fasta'}
    x_train, y_train, x_test, y_test = parse_metadata(**kwargs)

    if recount:
        allfiles = x_train + x_test
        count_kmers(k, l, allfiles, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(x_test, dtype='float64')

    return (x_train, y_train, x_test, y_test)


def get_kmer_us_uk_mixed(database="database", recount=False, k=7, l=13):
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
    kwargs = {'prefix': filepath, 'suffix': '.fasta', 'train_header': None}

    x_train, y_train, x_test, y_test = parse_metadata(**kwargs)

    if recount:
        genomes = x_train + x_test
        count_kmers(k, l, genomes, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(x_test, dtype='float64')

    return (x_train, y_train, x_test, y_test)


def get_genome_region_us_uk_mixed(table='Data/binary_table.txt', sep=None):
    """
    Parameters:
        table:   binary_table.txt output from panseq.
        threeD:  If True the data is made three dimensional.
    Returns:
        x_train, y_train, x_test, y_test

        binary genome region presence absence data ready to be input into a ml
        model, with us/uk data shuffled together.
    """
    kwargs = {'prefix': filepath, 'suffix': '.fasta', 'train_header': None}
    labels = parse_metadata(**kwargs)
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    x_train = []
    x_test = []
    if sep == None:
        data = pd.read_csv(table, sep=sep, engine='python', index_col=0)
    else:
        data = pd.read_csv(table, sep=sep, index_col=0)

    for header in x_train_labels:
        x_train.append(data[header].tolist())

    for header in x_test_labels:
        x_test.append(data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return (x_train, y_train, x_test, y_test)


def get_genome_region_us_uk_split(table='Data/binary_table.txt', sep=None):
    """
    Parameters:
        table:   binary_table.txt output from panseq.
        threeD:  If True the data is made three dimensional.
    Returns:
        x_train, y_train, x_test, y_test

        binary genome region presence absence data ready to be input into a ml
        model to recreate the Lupoloval et. al paper.
    """
    kwargs = {'prefix': filepath, 'suffix': '.fasta'}
    labels = parse_metadata(**kwargs)
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    x_train = []
    x_test = []
    if sep == None:
        data = pd.read_csv(table, sep=sep, engine='python', index_col=0)
    else:
        data = pd.read_csv(table, sep=sep, index_col=0)

    for header in x_train_labels:
        x_train.append(data[header].tolist())

    for header in x_test_labels:
        x_test.append(data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return (x_train, y_train, x_test, y_test)


def get_genome_custom_filtered(input_table='Data/binary_table.txt',
                               validation_table='Data/human_bovine_train_predictive.results',
                               sep=None,col='Ratio',cutoff=0.25,absolute=True,
                               greater=True,
                               args=None):
    """
    Parameters:
        input_table:        A binary_table output by panseq
        validation_table:   A table containing all the same rows as input_table,
                            but different columns.
        sep:                The delimiter used in input_table and
                            validation_table
        col:                The name of the column in validation_table that has
                            the values used to determine if a row will be kept.
        cutoff:             What the values in "col" are compared to to decide
                            if a row is kept or not.
        absolute:           If true the absolute value of the values in "col" is
                            used, if false the value is used as is
        greater:            If true the value in "col" must be greater than the
                            "cutoff" for a row to be kept, if false the value
                            must be less than "cutoff"
        args:               A list of arguments to be passed to parse_metadata.
    Returns:
        x_train, y_train, x_test, y_test ready to be input into a ml model, with
        all the rows that do not satisfy the constraints removed in
        validation_table removed.
    """
    if args:
        labels = parse_metadata(*args)
    else:
        labels = parse_metadata()
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    if sep == None:
        input_data=pd.read_csv(input_table,sep=sep,engine='python',index_col=0)
        validation_data=pd.read_csv(validation_table,sep=sep,engine='python',
                                    index_col=0)
    else:
        input_data = pd.read_csv(input_table, sep=sep, index_col=0)
        validation_data = pd.read_csv(validation_data, sep=sep, index_col=0)

    if absolute and greater:
        data = input_data[abs(validation_data[col]) > cutoff]
    elif absolute and not greater:
        data = input_data[abs(validation_data[col]) < cutoff]
    elif not absolute and greater:
        data = input_data[validation_data[col] > cutoff]
    elif not absolute and not greater:
        data = input_data[validation_data[col] < cutoff]

    x_train = []
    x_test = []

    for header in x_train_labels:
        x_train.append(data[header].tolist())

    for header in x_test_labels:
        x_test.append(data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return (x_train, y_train, x_test, y_test)

def get_genome_pre_filtered(input_table='Data/binary_table.txt',
                               validation_table='Data/human_bovine_train_predictive.results',
                               sep=None, count = 50,
                               args=None):
    """
    Parameters:
        input_table:        A binary_table output by panseq
        validation_table:   A table containing all the same rows as input_table,
                            but different columns.
        sep:                The delimiter used in input_table and
                            validation_table
        count:              How many of the top rows to keep.
        args:               A list of arguments to be passed to parse_metadata.
    Returns:
        x_train, y_train, x_test, y_test ready to be input into a ml model, with
        all the rows that do not satisfy the constraints removed in
        validation_table removed.
    """
    if args:
        labels = parse_metadata(*args)
    else:
        labels = parse_metadata()
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    if sep == None:
        input_data=pd.read_csv(input_table,sep=sep,engine='python',index_col=0)
        validation_data=pd.read_csv(validation_table,sep=sep,engine='python',
                                    index_col=0)
    else:
        input_data = pd.read_csv(input_table, sep=sep, index_col=0)
        validation_data = pd.read_csv(validation_data, sep=sep, index_col=0)

    validation_data = validation_data.head(count)
    input_data = input_data.loc[validation_data.index]

    x_train = []
    x_test = []

    for header in x_train_labels:
        x_train.append(input_data[header].tolist())

    for header in x_test_labels:
        x_test.append(input_data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return (x_train, y_train, x_test, y_test)


def get_kmer_from_json(database='database',recount=False,k=7,l=13,
                            path='/home/rboothman/moria/entero_db/',
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
    x_test = np.asarray(x_test, dtype='float64')

    scaler = MinMaxScaler(feature_range=(-1,1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, y_train, x_test, y_test)


def get_kmer_from_directory(database='database', recount=False, k=7, l=13,
                            threeD=True, scale=True, *directories):
    """
    Parameters:
        database,threeD,recount,k,l: See get_kmer_us_uk_split
        Scale:                       If true the data is scaled to range (-1,1)
                                     if false the data is not scaled.
        *directories:                One or more directories containing fasta
                                     files.
    Returns:
        A list of arrays of kmer count data. The list has the same number of
        elements as directories passed to the method. The first element of the
        list contains the kmer counts for the first directory passed into the
        method, the second element corresponds to the second directory etc. The
        scaling of the data will be done based on the first directory passed to
        the method. The files contained in  the directories will not be shuffled
    """
    all_files = []
    for directory in directories:
        all_files.append(setup_files(directory))

    if recount:
        count_kmers(k, l, all_files, database)

    output = []
    for directory in directories:
        temp = get_counts(setup_files(directory), database)
        temp = np.asarray(temp, dtype='float64')
        output.append(a)

    if scale:
        scaler = MinMaxScaler(feature_range(-1,1))
        temp = output
        output = []
        output.append(scaler.fit_transform(temp[0]))
        for array in temp[1:]:
            output.append(scaler.transform(array))

    if threeD:
        temp = output
        output = []
        for array in temp:
            array = make3D(array)
            output.append(array)

    return output

def get_salmonella_kmer(database='database2', recount=False, k=7, l=13,
                        antibiotic='ampicillin'):
    """
    Parmeters:
        database:   Name of the database to use.
        recount:    Boolean, if true the kmers are recounted.
        k:          The length of kmer to count.
        l:          The minimum number of times a kmer must appear to be output
        args:       Dictionary, the arguments to be passed to
                    parse_salmonella_metadata.
    Returns:
        (x_train,y_train,x_test,y_test): Kmer data ready to be passed to a
        machine learning model.
    """
    args = {'metadata': '/home/rboothman/PHAC/kmer/Data/amr_sorted.csv',
            'fasta_header': 'Fasta', 'label_header': 'AMR',
            'train_header': None, 'extra_header': 'Antibiotic',
            'extra_label': antibiotic,
            'prefix': '/home/rboothman/Data/salmonella_amr/', 'suffix': '.fna'}
    x_train, y_train, x_test, y_test = parse_metadata(**args)

    if recount:
        all_files = x_train + x_test
        count_kmers(k,l, all_files, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(x_test, dtype='float64')

    return (x_train, y_train, x_test, y_test)
