"""
A collection of methods that gather and prepare data to be input into a machine
learning model.

Most return: ((x_train, y_train, x_test, y_test), feature_names, test_files,
              LabelEncoder) where:
* x_train is a 2D array with shape (number of samples, number of features)
  containing the training data.
* y_train is a 1D array with shape (number of samples,) containing the
  classification labels for the training data.
* x_test is a 2D array of the shape (number of test samples, number of
  features) containing the test data_args.
* y_test is either a 1D array of the shape (number of test samples,) containing
  the classification labels for the test data or in the case where you are not
  validating the model is an empty array.
* feature_names is a list of all the features present in each sample.
* test_files is a list of the names of each input being used to test the model.
* LabelEncoder is a scikit-learn LabelEncoder object that will allow you to
  convert the predicted classifications back into a human readable format.
"""

from builtins import str
import logging
import os
from sklearn.preprocessing import Imputer
import kmerprediction.complete_kmer_counter as complete_kmer_counter
from kmerprediction.complete_kmer_counter import KmerCounterError
import kmerprediction.kmer_counter as kmer_counter
from kmerprediction.utils import shuffle, setup_files, parse_metadata, parse_json
from kmerprediction.utils import encode_labels
import numpy as np
import pandas as pd
from kmerprediction import constants


def get_kmer(metadata_kwargs=None, kmer_kwargs=None, recount=False,
             database=constants.DEFAULT_DB, validate=True,
             complete_count=True):
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
        validate (bool): If True y_test is created, if False y_test is
                         an empty ndarray.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    if complete_count:
        counter = complete_kmer_counter
    else:
        counter = kmer_counter

    metadata_kwargs = metadata_kwargs or {}
    metadata_kwargs['validate'] = validate
    kmer_kwargs = kmer_kwargs or {}

    if 'name' in kmer_kwargs:
        name = kmer_kwargs['name']
    else:
        name = constants.DEFAULT_NAME
    if 'output_db' in kmer_kwargs:
        output_db = kmer_kwargs['output_db']
    else:
        output_db = database

    (x_train, y_train, x_test, y_test) = parse_metadata(**metadata_kwargs)

    test_files = [str(x) for x in x_test]
    all_files = x_train + x_test

    if recount:
        counter.count_kmers(all_files, database, **kmer_kwargs, force=True)
    else:
        try:
            temp = counter.get_counts(x_train, output_db, name)
        except KmerCounterError as e:
            msg = 'Warning: get_counts failed, attempting a recount'
            logging.exception(msg)
            counter.count_kmers(all_files, database, **kmer_kwargs)

    x_train = counter.get_counts(x_train, output_db, name)
    x_test = counter.get_counts(x_test, output_db, name)

    feature_names = counter.get_kmer_names(output_db, name)

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
        validate (bool):    If True y_test is created, if False y_test is
                            an empty ndarray.

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


def get_kmer_us_uk_split(kmer_kwargs=None, database=constants.DEFAULT_DB,
                         recount=False, validate=True,
                         complete_count=True):
    """
    Wraps get_kmer to get the US/UK split dataset to recreate the Lupolova et
    al paper with kmer input data.

    Args:
        database (str):  lmdb database to store kmer counts.
        recount (bool):  If True the kmers are recounted.
        k (int):         Size of kmer to be counted. Ignored if recount is
                         False.
        L (int):         kmer cutoff value. Ignored if recount is false.
        validate (bool): Ignored, here for compatability.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    metadata_kwargs = {'prefix': constants.ECOLI,
                       'suffix': '.fasta',
                       'validate': validate}
    return get_kmer(metadata_kwargs=metadata_kwargs, kmer_kwargs=kmer_kwargs,
                    database=database, recount=recount, validate=validate,
                    complete_count=complete_count)


def get_kmer_us_uk_mixed(kmer_kwargs=None, database=constants.DEFAULT_DB,
                         recount=False, validate=True,
                         complete_count=True):
    """
    Wraps get_kmer to get the US/UK mixed dataset to recreate the Lupolova et
    al paper with kmer input data.

    Args:
        database (str): lmdb database to store kmer counts.
        recount (bool):  If True the kmers are recounted.
        k (int):         Size of kmer to be counted. Ignored if recount is
                         False.
        L (int):         kmer cutoff value. Ignored if recount is false.
        validate (bool): Ignored, here for compatability.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    metadata_kwargs = {'prefix': constants.ECOLI,
                       'suffix': '.fasta',
                       'train_header': None,
                       'validate': validate}
    return get_kmer(metadata_kwargs=metadata_kwargs, kmer_kwargs=kmer_kwargs,
                    database=database, recount=recount, validate=validate,
                    complete_count=complete_count)


def get_kmer_us_uk_reverse_split(kmer_kwargs=None, recount=False,
                                 validate=True, database=constants.DEFAULT_DB,
                                 complete_count=True):
    """
    Wraps get_kmer to get the US UK train test split reversed. I.e. The US set
    is the train set and the UK set is the test set.
    """
    metadata_kwargs = {'prefix': constants.ECOLI,
                       'suffix': '.fasta',
                       'train_header': 'Dataset',
                       'train_label': 'Test',
                       'test_label': 'Train',
                       'validate': validate}
    return get_kmer(metadata_kwargs=metadata_kwargs, kmer_kwargs=kmer_kwargs,
                    database=database, recount=recount, validate=validate,
                    complete_count=complete_count)


def get_kmer_us(kmer_kwargs=None, database=constants.DEFAULT_DB, recount=False,
                validate=True, complete_count=True):
    """
    Wraps get_kmer to get a random train/test split of just the US lupolova
    data.
    """
    metadata_kwargs = {'prefix': constants.ECOLI,
                       'suffix': '.fasta',
                       'train_header': None,
                       'extra_header': 'Dataset',
                       'extra_label': 'Test',
                       'validate': True}
    return get_kmer(metadata_kwargs=metadata_kwargs, kmer_kwargs=kmer_kwargs,
                    database=database, recount=recount, validate=validate,
                    complete_count=complete_count)


def get_kmer_uk(kmer_kwargs=None, database=constants.DEFAULT_DB, recount=False,
                validate=True, complete_count=True):
    """
    Wraps get_kmer to get a random train/test split of just the UK lupolova
    data.
    """
    metadata_kwargs = {'prefix': constants.ECOLI,
                       'suffix': '.fasta',
                       'train_header': None,
                       'extra_header': 'Dataset',
                       'extra_label': 'Train',
                       'validate': True}
    return get_kmer(metadata_kwargs=metadata_kwargs, kmer_kwargs=kmer_kwargs,
                    database=database, recount=recount, validate=validate,
                    complete_count=complete_count)


def get_salmonella_kmer(kmer_kwargs, antibiotic='ampicillin',
                        database=constants.DEFAULT_DB, recount=False,
                        validate=True, complete_count=True):
    """
    Wraps get_kmer to get salmonella amr data.

    Args:
        antibiotic (str): The anitibiotic to get amr data for.
        database (str):   lmdb database to store kmer counts.
        recount (bool):   If True the kmers are recounted.
        k (int):          Size of kmer to be counted. Ignored if recount is
                          false.
        L (int):          kmer cutoff value. Ignored if recount is false.
        validate (bool):  Ignored, here for compatability.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    metadata_kwargs = {'metadata': constants.SALMONELLA_METADATA,
                       'fasta_header': 'Fasta',
                       'label_header': 'AMR',
                       'train_header': None,
                       'extra_header': 'Antibiotic',
                       'extra_label': antibiotic,
                       'prefix': constants.SALMONELLA,
                       'suffix': '.fna',
                       'validate': True}
    return get_kmer(metadata_kwargs=metadata_kwargs, kmer_kwargs=kmer_kwargs,
                    database=database, recount=recount, validate=validate,
                    complete_count=complete_count)


def get_genome_region_us_uk_mixed(table=constants.GENOME_REGION_TABLE, sep=None,
                                  validate=True):
    """
    Wraps get_genome_regions to get the US/UK mixed datasets genome region data
    to recreate the Lupolova et al paper.

    Args:
        table (str):        binary_table.txt output from panseq.
        sep (str or None):  The separator used in table.
        validate (bool):    Ignored, here for compatability.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    kwargs = {'train_header': None,
              'validate': True}
    return get_genome_regions(kwargs, table, sep, validate=True)


def get_genome_region_us_uk_split(table=constants.GENOME_REGION_TABLE, sep=None,
                                  validate=True):
    """
    Wraps get_genome_regions to get the US/UK split dataset genome region data
    to recreate the Lupolova et al paper.

    Args:
        table (str):        binary_table.txt output from panseq.
        sep (str or None):  The separator used in table.
        validate (bool):    Ignored, here for compatability.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    kwargs = {'validate': True}
    return get_genome_regions(kwargs, table, sep, validate=True)


def get_omnilog_data(kwargs=None, omnilog_sheet=constants.OMNILOG_DATA,
                     validate=True):
    """
    Gets the omnilog data contained in omnilog_sheet for the genomes specified
    by kwargs. Uses utils.parse_metadata

    Args:
        kwargs (dict):       The arguments to pass to parse_metadata.
        omnilog_sheet (str): File containing omnilog data.
        validate (bool):     If True y_test is created, if False y_test is an
                             empty ndarray.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder

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


def get_roary_data(kwargs=None, roary_sheet=constants.ROARY, validate=True):
    """
    Get the Roary data from roary_sheet for the genomes specified by kwargs,
    uses utils.parse_metadata.

    Args:
        kwargs (dict):      The arguments to pass to parse_metadata.
        roary_sheet (str):  File containing Roary data.
        validate (bool):    If True y_test is created, if False y_test is an
                            empty ndarray.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
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
        validate (bool):    If True y_test is created, if False y_test is an
                            empty ndarray.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
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
        classes.append(roary_data[class_members].mean(axis=1) * 100)

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
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
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
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    kwargs = kwargs or {}

    labels = parse_metadata(**kwargs)
    train_label = labels[0]
    y_train = labels[1]
    test_label = labels[2]
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
        data = input_data.loc[filter_data.loc[abs(filter_data[col]) > cutoff].index]
    elif absolute and not greater:
        data = input_data.loc[filter_data.loc[abs(filter_data[col]) < cutoff].index]
    elif not absolute and greater:
        data = input_data.loc[filter_data.loc[filter_data[col] > cutoff].index]
    elif not absolute and not greater:
        data = input_data.loc[filter_data.loc[filter_data[col] < cutoff].index]

    x_train = []
    x_test = []

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
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    kwargs = kwargs or {}

    labels = parse_metadata(**kwargs)
    train_label = labels[0]
    y_train = labels[1]
    test_label = labels[2]
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

    for header in train_label:
        x_train.append(input_data[header].tolist())

    for header in test_label:
        x_test.append(input_data[header].tolist())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    feature_names = np.asarray(input_data.index)

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_label, le)


def get_kmer_from_json(train, test, database=constants.DEFAULT_DB,
                       recount=False, k=7, L=13, kwargs=None,
                       validate=True, complete_count=True):
    """
    Gets kmer data for the genomes specified in the json files. Divides genomes
    into train/test sets and classifies them with utils.parse_json.

    Args:
        train (str):        The filepath to the json file containing the
                            training genome inforamtion.
        test (str):         The filepath to the json file containing the testing
                            genome information.
        database (str):     lmdb database to store kmer counts.
        recount (bool):     If True the kmers are recounted.
        k (int):            Size of kmer to be counted. Ignored if recount is
                            false
        L (int):            kmer cutoff value. Ignored if recount is false.
        kwargs (dict):      The arguments to pass to parse_json.
        validate (bool):    If True and the kmers are being recounter a status
                            bar displaying the kmer count progress is output.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    if complete_count:
        counter = complete_kmer_counter
    else:
        counter = kmer_counter
    kwargs = kwargs or {}
    kwargs['validate'] = validate

    (x_train, y_train, x_test, y_test) = parse_json(train, test, **kwargs)

    test_files = [str(x) for x in x_test]

    if recount:
        counter.count_kmers(x_train + x_test, database, k=k, limit=L, force=True)

    x_train = counter.get_counts(x_train, database)
    x_test = counter.get_counts(x_test, database)

    feature_names = counter.get_kmer_names(database)

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_files, le)


def get_kmer_from_directory(train_dir, test_dir, database=constants.DEFAULT_DB,
                            recount=False, k=7, L=13, validate=True,
                            complete_count=True):
    """
    Organizes fasta files into train/test splits and classifies them based
    on their location in a directory structure rather than a metadata sheet.
    Returns kmer count data.

    With the following directory structure and train_dir set to Data/Train/ and
    test_dir set to Data/Test/ x_train would contain genomes 1-5 with genomes
    1-3 being classified as Class1 and genomes 4 and 5 being classified as
    Class2 (or whatever their respective directories are named) and x_test would
    contain genomes 6-9. If validate is true genomes 6 and 7 would be
    classified as Class1 and genmes 8 and 9 would classified as Class2. If
    validate is set to False genomes 6-9 will not be classified and y_test will
    be empty.

    Data/
      |--Train/
      |     |--Class1/
      |     |   |--genome1.fasta
      |     |   |--genome2.fasta
      |     |   |--genome3.fasta
      |     |
      |     |--Class2/
      |         |--genome4.fasta
      |         |--genome5.fasta
      |
      |--Test/
            |--Class1/
            |   |--genome6.fasta
            |   |--genome7.fasta
            |
            |--Class2/
                |--genome8.fasta
                |--genome9.fasta

    Args:
        train_dir (str):    Filepath to directory containing subdirectories for
                            each possible classification that contain fasta
                            files for training.
        test_dir (str):     Filepath to directory containing subdirectories for
                            each possible classification that contain fasta
                            files for testing.
        database (str):     lmdb database to store kmer counts.
        recount (bool):     If True the kmers are recounted.
        k (int):            Size of kmer to be counted. Ignored if recount is
                            false.
        L (int):            kmer cutoff value. Ignored if recount is false.
        validate (bool):    If True y_test is created, if False y_test is an
                            empty ndarray.

    Returns:
        tuple:  (x_train, y_train, x_test, y_test), feature_names, file_names,
                LabelEncoder
    """
    if complete_count:
        counter = complete_kmer_counter
    else:
        counter = kmer_counter

    train_directories = [train_dir + x for x in os.listdir(train_dir)]
    test_directories = [test_dir + x for x in os.listdir(test_dir)]

    train_files = []
    train_classes = []
    for d in train_directories:
        files = setup_files(d)
        train_files.append(files)
        train_classes.append(d.replace(train_dir, ''))

    test_files = []
    test_classes = []
    for d in test_directories:
        files = setup_files(d)
        test_files.append(files)
        test_classes.append(d.replace(test_dir, ''))

    if recount:
        all_files = train_files + test_files
        all_files = [x for l in all_files for x in l]
        counter.count_kmers(all_files, database, k=k, limit=L, force=True)

    train_counts = []
    for group in train_files:
        temp = counter.get_counts(group, database)
        temp = np.asarray(temp, dtype='float64')
        train_counts.append(temp)

    test_counts = []
    for group in test_files:
        temp = counter.get_counts(group, database)
        temp = np.asarray(temp, dtype='float64')
        test_counts.append(temp)

    test_files = [x for l in test_files for x in l]

    x_train, y_train = shuffle(train_counts, train_classes)
    x_test, y_test = shuffle(test_counts, test_classes)

    if not validate:
        y_test = np.array([], dtype='float64')

    feature_names = counter.get_kmer_names(database)

    y_train, y_test, le = encode_labels(y_train, y_test)

    output_data = (x_train, y_train, x_test, y_test)

    return (output_data, feature_names, test_files, le)
