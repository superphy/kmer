"""
Contains methods for gathering data and returning it in the form
x_train, y_train, x_test, y_test. Where x_train is the training data, y_train
is the labels for the training data, x_test is the testing data, and y_test is
the labels for the testing data.
"""

from sklearn.preprocessing import MinMaxScaler, Imputer
from kmer_counter import count_kmers, get_counts, get_kmer_names
from utils import shuffle, setup_files, parse_metadata, parse_json
from utils import encode_labels
import numpy as np
import pandas as pd
import constants


def get_kmer(kwargs=None, database=constants.DB, recount=False, k=7, L=13,
             validate=True):
    """
    Get kmer data for genomes specified in kwargs, uses kmer_counter and
    utils.parse_metadata

    Args:
        kwargs (dict):   The arguments to pass to parse_metadata
        database (str):  lmdb database to store kmer counts
        recount (bool):  If True the kmers are recounted
        k (int):         Size of kmer to be counted. Ignored if recount is
                         false
        L (int):         kmer cutoff value. Ignored if recount is false
        validate (bool): If True a list of the file names being predicted on is
                         returned

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """

    kwargs = kwargs or {}
    kwargs['validate'] = validate

    (x_train, y_train, x_test, y_test) = parse_metadata(**kwargs)

    test_files = [str(x) for x in x_test]

    if recount:
        count_kmers(k, L, x_train + x_test, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(x_test, dtype='float64')

    feature_names = get_kmer_names(database)

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_files, le)


def get_genome_regions(kwargs=None, table=constants.GENOME_REGION_TABLE,
                       sep=None, validate=True):
    """
    Gets genome region presence absence data from a binary table output by
    Panseq for the genomes specified by kwargs. Uses utils.parse_metadata

    Args:
        kwargs (dict):      The arguments to pass to parse_metadata.
        table (str):        binary_table.txt output from panseq.
        sep (str or None):  The separator used in table.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    kwargs = kwargs or {}
    kwargs['validate'] = validate

    (train_label, y_train, test_label, y_test) = parse_metadata(**kwargs)

    x_train = []
    x_test = []
    if sep is None:
        data = pd.read_csv(table, sep=sep, engine='python', index_col=0)
    else:
        data = pd.read_csv(table, sep=sep, index_col=0)

    for header in train_label:
        x_train.append(data[header].tolist())

    for header in test_label:
        x_test.append(data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    feature_names = np.asarray(data.index)

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_label, le)


def get_kmer_us_uk_split(database=constants.DB, recount=False, k=7, L=13):
    """
    Wraps get_kmer to get the US/UK split dataset to recreate the Lupolova et
    al paper with kmer input data.

    Args:
        database (str): lmdb database to store kmer counts.
        recount (bool): If True the kmers are recounted.
        k (int):        Size of kmer to be counted. Ignored if recount is false
        L (int):        kmer cutoff value. Ignored if recount is false.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, LabelEncoder
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'validate': True}
    return get_kmer(kwargs, database, recount, k, L, validate=True)


def get_kmer_us_uk_mixed(database=constants.DB, recount=False, k=7, L=13):
    """
    Wraps get_kmer to get the US/UK mixed dataset to recreate the Lupolova et
    al paper with kmer input data.

    Args:
        database (str): lmdb database to store kmer counts.
        recount (bool): If True the kmers are recounted.
        k (int):        Size of kmer to be counted. Ignored if recount is false
        L (int):        kmer cutoff value. Ignored if recount is false.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, LabelEncoder
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'train_header': None,
              'validate': True}
    return get_kmer(kwargs, database, recount, k, L, validate=True)


def get_salmonella_kmer(antibiotic='ampicillin', database=constants.DB,
                        recount=False, k=7, L=13):
    """
    Wraps get_kmer to get salmonella amr data.

    Args:
        antibiotic (str): The anitibiotic to get amr data for.
        database (str):   lmdb database to store kmer counts.
        recount (bool):   If True the kmers are recounted.
        k (int):          Size of kmer to be counted. Ignored if recount is
                          false.
        L (int):          kmer cutoff value. Ignored if recount is false.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, LabelEncoder
    """
    kwargs = {'metadata': constants.SALMONELLA_METADATA,
              'fasta_header': 'Fasta',
              'label_header': 'AMR',
              'train_header': None,
              'extra_header': 'Antibiotic',
              'extra_label': antibiotic,
              'prefix': constants.SALMONELLA,
              'suffix': '.fna',
              'validate': True}
    return get_kmer(kwargs, database, recount, k, L, validate=True)


def get_genome_region_us_uk_mixed(table=constants.GENOME_REGION_TABLE,
                                  sep=None):
    """
    Wraps get_genome_regions to get the US/UK mixed datasets genome region data
    to recreate the Lupolova et al paper.

    Args:
        table (str):        binary_table.txt output from panseq.
        sep (str or None):  The separator used in table.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, LabelEncoder
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'train_header': None,
              'validate': True}
    return get_genome_regions(kwargs, table, sep, validate=True)


def get_genome_region_us_uk_split(table=constants.GENOME_REGION_TABLE,
                                  sep=None):
    """
    Wraps get_genome_regions to get the US/UK split dataset genome region data
    to recreate the Lupolova et al paper.

    Args:
        table (str):        binary_table.txt output from panseq.
        sep (str or None):  The separator used in table.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, LabelEncoder
    """
    kwargs = {'prefix': constants.ECOLI,
              'suffix': '.fasta',
              'validate': True}
    return get_genome_regions(kwargs, table, sep, validate=True)


def get_genome_custom_filtered(input_table=constants.GENOME_REGION_TABLE,
                               filter_table=constants.PREDICTIVE_RESULTS,
                               sep=None, col='Ratio', cutoff=0.25,
                               absolute=True, greater=True, kwargs=None):
    """
    Gets genome region presence absence data from input_table, but performs
    initial feature selection using the values in col in filter_table. Uses
    utils.parse_metadata

    Args:
        input_table (str):  A binary_table output by panseq
        filter_table (str): A csv table to filter input_table by.
        sep (str):          The delimiter used in both tables.
        col (str):          Column name for the decision column in filter_table
        cutoff (float):     What the values in col are compared to,
        absolute (bool):    If true the absolute value of values in col is used
        greater (bool):     If true values in "col" must be greater than cutoff
        kwargs (dict):      Arguments to be passed to parse_metadata.

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """
    kwargs = kwargs or {}

    labels = parse_metadata(**kwargs)
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    if sep is None:
        input_data = pd.read_csv(input_table, sep=sep, engine='python',
                                 index_col=0)
        filter_data = pd.read_csv(filter_table, sep=sep, engine='python',
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
                           sep=None, count=50, kwargs=None):
    """
    Gets genome region presence absence from input_table for the genomes
    specified by kwargs. Does initial feature selection by using only the
    features in the top count rows of filter_table. Uses utils.parse_metadata

    Args:
        input_table (str):  A binary_table output by panseq
        filter_table (str): A table containing all the same rows as
                            input_table, but different columns.
        sep (str or None):  The delimiter used in input_table and filter_table
        count (int):        How many of the top rows to keep.
        kwargs (dict):      Arguments to be passed to parse_metadata.

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """
    kwargs = kwargs or {}

    labels = parse_metadata(**kwargs)
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    if sep is None:
        input_data = pd.read_csv(input_table, sep=sep, engine='python',
                                 index_col=0)
        validation_data = pd.read_csv(filter_table, sep=sep, engine='python',
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


# TODO: convert *json to (*train_json, *test_json) where each json file
# TODO: contains genomes from only one class.
def get_kmer_from_json(database=constants.DB, recount=False, k=7, L=13,
                       prefix=constants.MORIA, suffix='.fasta',
                       key='assembly_barcode', *json):
    """
    Gets kmer data for the genomes specified in the json files. Divides genomes
    into train and test sets passed on which json file they are specified in.
    Uses parse_json

    Args:
        database (str): lmdb database to store kmer counts.
        recount (bool): If True the kmers are recounted.
        k (int):        Size of kmer to be counted. Ignored if recount is false
        L (int):        kmer cutoff value. Ignored if recount is false.
        prefix (str):   String to be added to the beginning of every fasta file
                        in the json, for example the correct path to the file.
        suffix (str):   Suffix to be added to the end of every fasta file in
                        the json, for example .fasta
        *json (str):    Two or four json files, If two the first should contain
                        only positive genomes and the second should contain
                        only negative genomes. If four the first json file
                        should contain all positive training genomes, the
                        second all negative training genomes, the third all
                        positive test genomes, the fourth all negative test
                        genomes.

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """
    files = parse_json(prefix, suffix, key, *json)

    if len(files) == 4:
        x_train, y_train = shuffle(files[0] + files[1], [1, 0])
        x_test, y_test = shuffle(files[2] + files[3], [1, 0])
    elif len(files) == 2:
        x_train, y_train = shuffle(files[0] + files[1], [1, 0])
        cutoff = int(0.8*len(x_train))
        x_test = x_train[cutoff:]
        y_test = y_train[cutoff:]
        x_train = x_train[:cutoff]
        y_train = y_train[:cutoff]

    if recount:
        all_files = x_train + x_test
        count_kmers(k, L, all_files, database)

    x_train = get_counts(x_train, database)
    x_train = np.asarray(x_train, dtype='float64')

    x_test = get_counts(x_test, database)
    x_test = np.asarray(x_test, dtype='float64')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, y_train, x_test, y_test)


# TODO: Convert this to work with run
# TODO: i.e. needs to return output_data,feature_names,test_files,labelencoder
def get_kmer_from_directory(database=constants.DB, recount=False, k=7, L=13,
                            *directories):
    """
    Gets kmer data from the fasta files in *directories. Does not work with
    run.py instead of separating output into train and test sets, the genomes
    and grouped based on the directory they were stored in.

    Args:
        database (str):     lmdb database to store kmer counts.
        recount (bool):     If True the kmers are recounted.
        k (int):            Size of kmer to be counted. Ignored if recount is
                            false.
        L (int):            kmer cutoff value. Ignored if recount is false.
        *directories (str): One or more directories containing fasta files.

    Returns:
        tuple: Has as many items as *directories, each element is an array of
               kmer count data.
    """
    all_files = []
    for directory in directories:
        all_files.append(setup_files(directory))

    if recount:
        count_kmers(k, L, all_files, database)

    output = []
    for directory in directories:
        temp = get_counts(setup_files(directory), database)
        temp = np.asarray(temp, dtype='float64')
        output.append(temp)

    return output


# TODO: Update docstring
def get_omnilog_data(kwargs=None, omnilog_sheet=constants.OMNILOG_DATA,
                     validate=True):
    """
    Gets the omnilog data contained in omnilog_sheet for the genomes specified
    by kwargs. Uses utils.parse_metadata

    Args:
        kwargs (dict):       The arguments to pass to parse_metadata.
        omnilog_sheet (str): File containing omnilog data.
        validate (bool):     If True a list of the file names being predicted
                             on is returned

    Returns:
        list: x_train, y_train, x_test, y_test
        or
        tuple: ([x_train, y_train, x_test, y_test], feature_names)

    """
    kwargs = kwargs or {}
    kwargs['validate'] = validate

    (x_train, y_train, x_test, y_test) = parse_metadata(**kwargs)

    test_files = [str(x) for x in x_test]

    omnilog_data = pd.read_csv(omnilog_sheet, index_col=0)
    valid_cols = [x_train.index(x) for x in x_train if x in list(omnilog_data)]
    x_train = [x_train[x] for x in valid_cols]
    y_train = [y_train[x] for x in valid_cols]

    valid_cols = [x_test.index(x) for x in x_test if x in list(omnilog_data)]
    x_test = [x_test[x] for x in valid_cols]
    if validate:
        y_test = [y_test[x] for x in valid_cols]

    feature_names = omnilog_data.index

    output_data = []
    x_train = omnilog_data[x_train].T.values
    x_test = omnilog_data[x_test].T.values

    imputer = Imputer()
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return output_data, feature_names, test_files, le


# TODO: Update docstring
def get_roary_data(kwargs=None, roary_sheet=constants.ROARY, validate=True):
    """
    Get the Roary data from roary_sheet for the genomes specified by kwargs,
    uses utils.parse_metadata.

    Args:
        kwargs (dict):      The arguments to pass to parse_metadata.
        roary_sheet (str):  File containing Roary data.

    Returns:
        list: x_train, y_train, x_test, y_test
    """
    kwargs = kwargs or {}
    kwargs['validate'] = validate

    (x_train, y_train, x_test, y_test) = parse_metadata(**kwargs)

    test_files = [str(x) for x in x_test]

    roary_data = pd.read_csv(roary_sheet, index_col=0)

    feature_names = roary_data.index

    valid_cols = [x_train.index(x) for x in x_train if x in list(roary_data)]
    x_train = [x_train[x] for x in valid_cols]
    y_train = [y_train[x] for x in valid_cols]

    valid_cols = [x_test.index(x) for x in x_test if x in list(roary_data)]
    x_test = [x_test[x] for x in valid_cols]
    y_test = [y_test[x] for x in valid_cols]

    x_train = roary_data[x_train].T.values
    x_test = roary_data[x_test].T.values

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_files, le)


# TODO: Update docstring
def get_filtered_roary_data(kwargs=None, roary_sheet=constants.ROARY, limit=10,
                            validate=True):
    """
    Gets the Roary data from roary_sheet for the genomes specified by kwargs,
    uses utils.parse_metadata. Does initial feature selection by removing
    features whose in proportion between classes is less than limit, based on
    the feature selection done by Lupolova et. al.

    Args:
        kwargs (dict):      The arguments to pass to parse_metadata.
        roary_sheet (str):  File containing Roary data.
        limit (int):        Value used to determine which features are removed

    Returns:
        list: x_train, y_train, x_test, y_test
    """
    kwargs = kwargs or {}
    kwargs['validate'] = validate

    (x_train, y_train, x_test, y_test) = parse_metadata(**kwargs)

    test_files = [str(x) for x in x_test]

    roary_data = pd.read_csv(roary_sheet, index_col=0)

    class_labels = np.unique(y_train)
    classes = []
    for c in class_labels:
        class_members = [x for x in x_train if y_train[x_train.index(x)] == c]
        print roary_data[class_members].mean(axis=1)*100
        exit()
        classes.append(roary_data[class_members].mean(axis=1)*100)

    proportions = pd.concat(classes, axis=1)
    diffs = np.diff(proportions.values, axis=1)
    diffs = np.absolute(diffs.mean(axis=1))
    idx = list(proportions.index)
    col = ['Diff']
    avg_diff = pd.DataFrame(diffs, index=idx, columns=col)
    invalid = list(avg_diff[avg_diff['Diff'] < limit].index)
    roary_data = roary_data.drop(invalid)

    feature_names = roary_data.index

    valid_cols = [x_train.index(x) for x in x_train if x in list(roary_data)]
    x_train = [x_train[x] for x in valid_cols]
    y_train = [y_train[x] for x in valid_cols]

    valid_cols = [x_test.index(x) for x in x_test if x in list(roary_data)]
    x_test = [x_test[x] for x in valid_cols]
    if validate:
        y_test = [y_test[x] for x in valid_cols]

    x_train = roary_data[x_train].T.values
    x_test = roary_data[x_test].T.values

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_files, le)


# TODO: Update docstring
def get_roary_from_list(kwargs=None, roary_sheet=constants.ROARY,
                        gene_header='Gene', valid_header='Valid',
                        valid_features_table=constants.ROARY_VALID):
    """
    Gets the Roary data from roary_sheet for the genomes specified by kwargs,
    uses utils.parse_metadata. Does initial feature selection by removing
    features who are not labeled as valid in valid_features_table.

    Args:
        kwargs (dict):              The arguments to pass to parse_metadata.
        roary_sheet (str):          File containing Roary data.
        gene_header (str):          Header for the column that contains the
                                    gene names.
        valid_header (str):         Header for the column that contains T/F
                                    values determining if a gene is valid.
        valid_features_table (str): csv table containing a list of valid and
                                    invalid genes.

    Returns:
        list: x_train, y_train, x_test, y_test
    """
    kwargs = kwargs or {}

    (x_train, y_train, x_test, y_test) = parse_metadata(**kwargs)

    test_files = [str(x) for x in x_test]

    roary_data = pd.read_csv(roary_sheet)
    valid_features = pd.read_csv(valid_features_table)
    features = list(valid_features[valid_header])
    roary_data = roary_data[roary_data[gene_header].isin(features)]

    valid_cols = [x_train.index(x) for x in x_train if x in list(roary_data)]
    x_train = [x_train[x] for x in valid_cols]
    y_train = [y_train[x] for x in valid_cols]

    valid_cols = [x_test.index(x) for x in x_test if x in list(roary_data)]
    x_test = [x_test[x] for x in valid_cols]
    if list(y_test):
        y_test = [y_test[x] for x in valid_cols]

    x_train = roary_data[x_train].T.values
    x_test = roary_data[x_test].T.values

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, features, test_files, le)
