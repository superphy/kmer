from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from kmer_counter import count_kmers, get_counts, get_kmer_names
from utils import check_fasta, same_shuffle, shuffle, flatten, make3D, setup_files
from utils import parse_metadata, parse_json
import numpy as np
import pandas as pd
import csv
import os
import random
import constants


def get_kmer(kwargs={}, database=constants.DB, recount=False, k=7, l=13,
             extract=False):
    """
    Args:
        args:       The arguments to pass to parse_metadata.
        database:   lmdb database to store kmer counts.
        threeD:     If True the data is made three dimensional.
        recount:    If True the kmers are recounted.
        k:          Size of kmer to be counted. Ignored if recount is false.
        l:          kmer cutoff value. Ignored if recount is false.

    Returns:
        (x_train, y_train, x_test, y_test) ready to be passed to a ml model
    """
    x_train, y_train, x_test, y_test = parse_metadata(**kwargs)

    if recount:
        allfiles = x_train + x_test
        count_kmers(k, l, allfiles, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(x_test, dtype='float64')

    output_data = (x_train, y_train, x_test, y_test)

    if extract:
        feature_names = get_kmer_names(database)
        output = (output_data, feature_names)
    else:
        output = output_data

    return output


def get_genome_regions(kwargs={},table=constants.GENOME_REGION_TABLE,sep=None,
                      extract=False):
    """
    Args:
        args:    The arguments to pass to parse_metadata.
        table:   binary_table.txt output from panseq.

    Returns:
        x_train, y_train, x_test, y_test
    """
    x_train_labels, y_train, x_test_labels, y_test = parse_metadata(**kwargs)

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

    output_data = (x_train, y_train, x_test, y_test)

    if extract:
        feature_names = np.asarray(data.index)
        output = (output_data, feature_names)
    else:
        output = output_data

    return output


def get_kmer_us_uk_split(database=constants.DB, recount=False, k=7, l=13,
                         extract=False):
    """
    Wraps get_kmer to get the US/UK split dataset kmer data to recreate the
    Lupolova et al paper.
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta'}
    return get_kmer(kwargs, database, recount, k, l, extract)


def get_kmer_us_uk_mixed(database=constants.DB, recount=False, k=7, l=13,
                         extract_feature=False):
    """
    Wraps get_kmer to get the US/UK mixed dataset kmer data to recreate the
    Lupolova et al paper.
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'train_header': None}
    return get_kmer(kwargs, database, recount, k, l, extract)


def get_salmonella_kmer(antibiotic='ampicillin', database=constants.DB,
                        recount=False, k=7, l=13, extract=False):
    """
    Wraps get_kmer to get salmonella amr data.
    """
    kwargs = {'metadata': constants.SALMONELLA_METADATA,
              'fasta_header': 'Fasta',
              'label_header': 'AMR',
              'train_header': None,
              'extra_header': 'Antibiotic',
              'extra_label': antibiotic,
              'prefix': constants.SALMONELLA,
              'suffix': '.fna'}
    return get_kmer(kwargs, database, recount, k, l, extract)


# def get_omnilog_kmer(database=constants.DB, recount=False, k=7, l=13,
#                      label_header='Host', one_vs_all=None,
#                      extract=False):
#     """
#     Wraps get_kmer, to get kmer data for the omnilog fasta files.
#     """
#     kwargs = {'metadata': 'Data/metadata.csv',
#               'fasta_header': 'Strain',
#               'label_header': classification_header,
#               'train_header': None,
#               'one_vs_all': positive_label,
#               'prefix': constants.OMNILOG_FASTA,
#               'suffix': '.fasta'}
#     return get_kmer(kwargs, database, recount, k, l, extract)


def get_genome_region_us_uk_mixed(table=constants.GENOME_REGION_TABLE,sep=None,
                                  extract=False):
    """
    Wraps get_genome_regions to get the US/UK mixed datasets genome region data to
    recreate the Lupolova et al paper.
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'train_header': None}
    return get_genome_regions(kwargs, table, sep, extract)


def get_genome_region_us_uk_split(table=constants.GENOME_REGION_TABLE,sep=None,
                                  extract=False):
    """
    Wraps get_genome_regions to get the US/UK split dataset genome region data to
    recreate the Lupolova et al paper.
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta'}
    return get_genome_regions(kwargs, table, sep, extract)


def get_genome_custom_filtered(input_table=constants.GENOME_REGION_TABLE,
                               filter_table=constants.PREDICTIVE_RESULTS,
                               sep=None,col='Ratio',cutoff=0.25,absolute=True,
                               greater=True,kwargs=None):
    """
    Args:
        input_table:        A binary_table output by panseq
        filter_table:       A table to filter input_table by.
        sep:                The delimiter used in both tables.
        col:                Column name for the decision column in filter_table.
        cutoff:             Float, what the values in col are compared to,
        absolute:           If true the absolute value of values in col is used.
        greater:            If true values in "col" must be greater than cutoff.
        kwargs:             Dictionary of arguments for parse_metadata.
    Returns:
        x_train, y_train, x_test, y_test ready to be input into a ml model
    """
    if args:
        labels = parse_metadata(**kwargs)
    else:
        labels = parse_metadata()
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    if sep == None:
        input_data=pd.read_csv(input_table,sep=sep,engine='python',index_col=0)
        filter_data=pd.read_csv(filter_table,sep=sep,engine='python',
                                index_col=0)
    else:
        input_data = pd.read_csv(input_table, sep=sep, index_col=0)
        filter_data = pd.read_csv(filter_table, sep=sep, index_col=0)

    if absolute and greater:
        data = input_data[abs(filter_data[col]) > cutoff]
    elif absolute and not greater:
        data = input_data[abs(filter_data[col]) < cutoff]
    elif not absolute and greater:
        data = input_data[filter_data[col] > cutoff]
    elif not absolute and not greater:
        data = input_data[filter_data[col] < cutoff]

    x_train = []
    x_test = []

    for header in x_train_labels:
        x_train.append(data[header].tolist())

    for header in x_test_labels:
        x_test.append(data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return (x_train, y_train, x_test, y_test)

def get_genome_prefiltered(input_table=constants.GENOME_REGION_TABLE,
                            filter_table=constants.PREDICTIVE_RESULTS,
                            sep=None, count=50, args=None):
    """
    Args:
        input_table:        A binary_table output by panseq
        filter_table:       A table containing all the same rows as input_table,
                            but different columns.
        sep:                The delimiter used in input_table and
                            filter_table
        count:              How many of the top rows to keep.
        args:               A list of arguments to be passed to parse_metadata.
    Returns:
        x_train, y_train, x_test, y_test ready to be input into a ml model, with
        all the rows that do not satisfy the constraints removed in
        filter_table removed.
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
        validation_data=pd.read_csv(filter_table,sep=sep,engine='python',
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


def get_kmer_from_json(database=constants.DB,recount=False,k=7,l=13,
                       path=constants.MORIA,suffix='.fasta',
                       key='assembly_barcode',*json):
    """
    Args:
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


def get_kmer_from_directory(database=constants.DB, recount=False, k=7, l=13,
                            threeD=True, scale=True, *directories):
    """
    Args:
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

def get_omnilog_data(kwargs={}, omnilog_sheet=constants.OMNILOG_DATA,
                     extract=False):
    """

    """
    input_data = list(parse_metadata(**kwargs))
    omnilog_data = pd.read_csv(omnilog_sheet, index_col=0)
    valid_cols = [input_data[0].index(x) for x in input_data[0] if x in list(omnilog_data)]
    input_data[0] = [input_data[0][x] for x in valid_cols]
    input_data[1] = [input_data[1][x] for x in valid_cols]
    valid_cols = [input_data[2].index(x) for x in input_data[2] if x in list(omnilog_data)]
    input_data[2] = [input_data[2][x] for x in valid_cols]
    input_data[3] = [input_data[3][x] for x in valid_cols]

    output_data = []
    output_data.append(omnilog_data[input_data[0]].T.values)
    output_data.append(input_data[1])
    output_data.append(omnilog_data[input_data[2]].T.values)
    output_data.append(input_data[3])

    imputer = Imputer()
    output_data[0] = imputer.fit_transform(output_data[0])
    output_data[2] = imputer.transform(output_data[2])
    if extract:
        output = (output_data, np.asarray(omnilog_data.index))
    else:
        output = output_data
    return output

def get_roary_data(kwargs={}, roary_sheet=constants.ROARY):
    input_data = list(parse_metadata(**kwargs))
    roary_data = pd.read_csv(roary_sheet,index_col=0)

    valid_cols = [input_data[0].index(x) for x in input_data[0] if x in list(roary_data)]
    input_data[0] = [input_data[0][x] for x in valid_cols]
    input_data[1] = [input_data[1][x] for x in valid_cols]

    valid_cols = [input_data[2].index(x) for x in input_data[2] if x in list(roary_data)]
    input_data[2] = [input_data[2][x] for x in valid_cols]
    input_data[3] = [input_data[3][x] for x in valid_cols]

    output_data = []
    output_data.append(roary_data[input_data[0]].T.values)
    output_data.append(input_data[1])
    output_data.append(roary_data[input_data[2]].T.values)
    output_data.append(input_data[3])

    return output_data

def get_filtered_roary_data(kwargs={}, roary_sheet=constants.ROARY, limit=10):
    input_data = list(parse_metadata(**kwargs))

    roary_data = pd.read_csv(roary_sheet, index_col=0)

    class_labels = np.unique(input_data[1])
    classes = []
    for c in class_labels:
        class_members = [x for x in input_data[0] if input_data[1][input_data[0].index(x)]==c]
        print roary_data[class_members].mean(axis=1)*100
        exit()
        classes.append(roary_data[class_members].mean(axis=1)*100)

    proportions = pd.concat(proportions, axis=1)
    diffs = np.diff(proportions.values, axis=1)
    idx = list(proportions.index)
    col = ['Diff']
    avg_diff=pd.DataFrame(np.absolute(diffs.mean(axis=1)),index=idx,columns=col)
    invalid = list(avg_diff[avg_diff['Diff'] < limit].index)
    roary_data = roary_data.drop(invalid)

    valid_cols = [input_data[0].index(x) for x in input_data[0] if x in list(roary_data)]
    input_data[0] = [input_data[0][x] for x in valid_cols]
    input_data[1] = [input_data[1][x] for x in valid_cols]

    valid_cols = [input_data[2].index(x) for x in input_data[2] if x in list(roary_data)]
    input_data[2] = [input_data[2][x] for x in valid_cols]
    input_data[3] = [input_data[3][x] for x in valid_cols]

    output_data = []
    output_data.append(roary_data[input_data[0]].T.values)
    output_data.append(input_data[1])
    output_data.append(roary_data[input_data[2]].T.values)
    output_data.append(input_data[3])

    return output_data

def get_roary_from_list(kwargs={},roary_sheet=constants.ROARY,
                        gene_header='Gene',valid_header='Valid',
                        valid_features_table=constants.ROARY_VALID):
    input_data = list(parse_metadata(**kwargs))

    roary_data = pd.read_csv(roary_sheet)
    valid_features = pd.read_csv(valid_features_table)
    features = list(valid_features[valid_header])
    roary_data = roary_data[roary_data[gene_header].isin(features)]

    valid_cols = [input_data[0].index(x) for x in input_data[0] if x in list(roary_data)]
    input_data[0] = [input_data[0][x] for x in valid_cols]
    input_data[1] = [input_data[1][x] for x in valid_cols]

    valid_cols = [input_data[2].index(x) for x in input_data[2] if x in list(roary_data)]
    input_data[2] = [input_data[2][x] for x in valid_cols]
    input_data[3] = [input_data[3][x] for x in valid_cols]

    output_data = []
    output_data.append(roary_data[input_data[0]].T.values)
    output_data.append(input_data[1])
    output_data.append(roary_data[input_data[2]].T.values)
    output_data.append(input_data[3])

    return output_data
