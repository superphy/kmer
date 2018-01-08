"""
Odds and ends used by other modules in the program.
"""

import os
import random
import json
import re
import pandas as pd
import numpy as np
import constants
from Bio import SeqIO


def setup_files(filepath):
    """
    Takes a path to a directory, returns a list of the complete paths to each
    file in the directory

    Args:
        filepath (str): Path to directory of files.

    Returns:
        list(str): Complete path to every file in filepath.
    """
    if not filepath[-1] == '/':
        filepath += '/'
    return [filepath + x for x in os.listdir(filepath)]


def check_fasta(filename):
    """
    Returns True if the file is fasta or fastq, False otherwise.

    Args:
        filename (str): File to check.

    Returns:
        bool: True if file is fasta|q otherwise False.
    """
    with open(filename, 'r') as f:
        fasta = SeqIO.parse(f, "fasta")
        fastq = SeqIO.parse(f, "fastq")
        return bool(any(fasta) or any(fastq))


def valid_file(test_files, *invalid_files):
    """
    Checks a list of files against csv files that contain invalid file names,
    removes any files from the input list that are in the csv files.

    Args:
        test_files (list(str)): Filenames to check.
        *invalid_files (str):   One or more csv files that contain a list of
                                invalid fasta file names.
    Returns:
        list(str): list of file names with all invalid ones remvoved.
    """
    bad_files = []
    for f in invalid_files:
        with open(f, 'r') as temp:
            bad_files.extend(temp.read().split('\n'))
    return [x for x in test_files if x not in bad_files]


def same_shuffle(a, b):
    """
    Shuffles two lists so that the elements at index x in both lists before
    shuffling are at index y in their respective list after shuffling.

    Args:
        a (list): A list of elements to shuffle.
        b (list): Another list of elements to shuffle, should have same length
                  as a.

    Returns:
        tuple: a,b with their elements shuffled.
    """
    temp = list(zip(a, b))
    random.shuffle(temp)
    a, b = zip(*temp)
    return list(a), list(b)


def shuffle(data, labels):
    """
    Args:
        data (list):
        labels (list):

    Returns:
        tuple: all_data, all_labels
    """
    try:
        assert len(data) == len(labels)
        assert isinstance(data[0], (list, np.ndarray))
    except AssertionError as E:
        print E
    if isinstance(data[0], list):
        all_data = []
        for x in data:
            all_data.extend(x)
        all_labels = []
        count = 0
        for label in labels:
            all_labels.extend([label for x in data[count]])
            count += 1
        all_data, all_labels = same_shuffle(all_data, all_labels)
    elif type(data[0] == np.ndarray):
        all_data = np.concatenate(data, axis=0)
        all_labels = []
        count = 0
        for label in labels:
            all_labels.append(np.full(data[count].shape[0], label))
            count += 1
        all_labels = np.concatenate(all_labels, axis=0)
        all_data, all_labels = same_shuffle(all_data, all_labels)
        all_data = np.asarray(all_data)
        all_labels = np.asarray(all_labels)
    return all_data, all_labels


def flatten(data):
    """
    Takes a 3D numpy ndarray and makes it 2D, assumes that the array was made 3D
    using utils.make3D

    Args:
        data (ndarray): Numpy array to flatten.

    Returns:
        ndarray: The input array with its inner dimension removed.
    """
    data = data.reshape(data.shape[0], data.shape[1])
    return data


def make3D(data):
    """
    Takes a 2D numpy ndarray and makes it 3D to be used in a Conv1D keras layer.

    Args:
        data (ndarray): Numpy array to make 3D

    Returns:
        ndarray: The input array with an additional dimension added.
    """
    data = data.reshape(data.shape[0], data.shape[1], 1)
    return data


def sensitivity_specificity(predicted_values, true_values):
    """
    Calculates the sensitivity and specificty of a machine learning model.

    Args:
        predicted_values:   What the model predicted.
        true_values:        The true values.
    Returns:
        dict(dict): Outer dictionary has keys for each class in the data, inner
                    dictionary has keys for sensitivity and specificity.
    """
    results = {}
    if isinstance(predicted_values, np.ndarray):
        predicted_values = np.asarray(predicted_values)
    if isinstance(true_values, np.ndarray):
        true_values = np.asarray(true_values)

    classes = np.unique(true_values)
    for c in classes:
        predicted_pos = np.where(predicted_values == c, 1, 0)
        predicted_neg = np.where(predicted_values != c, 1, 0)
        true_pos = np.where(true_values == c, 1, 0)
        true_neg = np.where(true_values != c, 1, 0)

        TP = sum(true_pos)
        TN = sum(true_neg)
        FP = sum(true_neg & predicted_pos)
        FN = sum(true_pos & predicted_neg)

        sensitivity = (1.0*TP)/(TP+FN)
        specificity = (1.0*TN)/(TN+FP)

        results[c] = {'sensitivity':sensitivity, 'specificity':specificity}

    return results


def parse_metadata(metadata=constants.ECOLI_METADATA, fasta_header='Fasta',
                   label_header='Class', train_header='Dataset',
                   extra_header=None, extra_label=None, train_label='Train',
                   test_label='Test', suffix='', prefix='', sep=None,
                   one_vs_all=None, remove=None, validate=True, blacklist=None):
    """
    Gets filenames, classifications, and train/test splits from a metadata
    sheet. Does not alter the metadata sheet in any way. Provides options to
    not use specific genomes that are present in the metadata sheet.

    Args:
        metadata (str):         A csv file, must contain at least one column of
                                genome names and one column of their
                                classifications.
        fasta_header (str):     Header for the genome name column
        label_header (str):     Header for the genome classification column
        train_header (str):     Header for the column that contains train/test
                                labels, if not given a random 80/20 split will
                                be used to generate the train/test datasets.
        extra_header (str):     Header for an additional column.
        extra_label (str):      If a sample's value under the extra_header
                                column does not match extra_label it will be
                                removed.
        train_label (str):      Labels for train genomes under train_header.
        test_label (str):       Labels for test genomes under train_header.
        suffix (str):           Suffix to appened to the end of genome names,
                                for example .fasta
        prefix (str):           Prefix to attach to the front of genome names,
                                for example the complete filepath.
        sep (str):              The delimiter used in metadata, if None the
                                delimiter is guessed.
        one_vs_all (str):       If given, changes a multiclass problem into a
                                binary one. All samples whose classification
                                does not match the given value will be combined
                                to form one class.
        remove (str):           If given, any samples whose classification
                                matches the given value will be removed.
        validate (bool):        If True y_test is created, if False y_test is an
                                empty ndarray.
        blacklist (list(str)):  A list of genome names to remove.

    Returns:
        tuple: (x_train, y_train, x_test, y_test); x_train and x_test contain
               filenames, not the actual data to be passed to a machine learning
               model.
    """
    if sep is None:
        data = pd.read_csv(metadata, sep=sep, engine='python')
    else:
        data = pd.read_csv(metadata, sep=sep)

    if extra_header:
        data = data[data[extra_header] == extra_label]

    if remove:
        data = data.drop(data[data[label_header] == remove].index)

    if blacklist is not None:
        data = data.drop(data[data[fasta_header].isin(blacklist)].index)

    if one_vs_all:
        data[label_header] = data[label_header].where(data[label_header] == one_vs_all, 'Other')

    all_labels = np.unique(data[label_header])
    all_labels = all_labels[pd.notnull(all_labels)]

    if train_header:
        train_data = data[data[train_header] == train_label]
        test_data = data[data[train_header] == test_label]
        all_train_data = []
        all_test_data = []
        for label in all_labels:
            all_train_data.append(train_data[train_data[label_header] == label])
            if validate:
                all_test_data.append(test_data[test_data[label_header] == label])
        all_train_data = [x[fasta_header].values for x in all_train_data]
        if validate:
            all_test_data = [x[fasta_header].values for x in all_test_data]
        else:
            all_test_data = test_data[fasta_header].values
    else:
        all_train_data = []
        all_test_data = []
        for label in all_labels:
            label_data = data[data[label_header] == label]
            label_data = label_data[fasta_header].values
            np.random.shuffle(label_data)
            if label_data.shape[0] == 1:
                all_train_data.append(label_data[0:])
                all_test_data.append(label_data[:0])
            else:
                cutoff = int(0.8*label_data.shape[0])
                all_train_data.append(label_data[:cutoff])
                all_test_data.append(label_data[cutoff:])
    all_train_data = [[prefix+str(x)+suffix for x in array] for array in all_train_data]
    x_train, y_train = shuffle(all_train_data, all_labels)
    if validate:
        all_test_data = [[prefix+str(x)+suffix for x in array] for array in all_test_data]
        x_test, y_test = shuffle(all_test_data, all_labels)
    else:
        x_test = [prefix+str(x)+suffix for x in all_test_data]
        y_test = np.array([], dtype='float64')

    return (x_train, y_train, x_test, y_test)


def parse_json(json_files, path=constants.MORIA, suffix='.fasta',
               key='assembly_barcode'):
    """
    Args:
        path:       File path to be appended to the beginning of each fasta file
        suffix:     String to be appended to the end of each fasta file eg
                    ".fasta"
        key:        The fasta filename identifier used in the json files.
        json_files: One or more json files to create a list of fasta files from.
    Returns:
        A list with as many elements as json files were input. Each element in
        the list is a list of the complete file paths to each valid genome
        contained in the corresponding json file.
    See Superphy/MoreSerotype/module/DownloadMetadata.py on Github for a script
    that can generate the json files.
    """
    output = []
    for f in json_files:
        with open(f, 'r') as f:
            data = json.load(f)
            fasta_names = [str(x[key]) for x in data]
            fasta_names = [x for x in fasta_names if valid_file(x)]
            fasta_names = [path + x + suffix for x in fasta_names]
            fasta_names = [x for x in fasta_names if check_fasta(x)]
        output.append(fasta_names)

    return output


def convert_well_index(well_index):
    """
    Converts the omnilog well coordinates to what was actually in the well.

    Args:
        well_index (str): The well index in form PM(number)...(Letter)(Number)

    Returns:
        str: The input well index simplified to PM(number)-(Letter)(number)
             followed by what was in the well.
    """
    well_descriptions = pd.read_csv(constants.OMNILOG_WELLS)
    first = re.compile(r'^PM\d+')
    second = re.compile(r'[A-H]\d+$')
    first_result = re.search(first, well_index)
    second_result = re.search(second, well_index)
    if first_result and second_result:
        df_index = first_result.group(0) + '-' + second_result.group(0)
        output = well_descriptions.loc[well_descriptions['Key'] == df_index]
        output = output.Key.item() + output.Value.item()
    else:
        output = well_index
    return output


def do_nothing(input_data, **kwargs):
    """
    A method that does nothing. Takes an input and returns it, also returns
    feature names if it is passed though kwargs. Used to remove complex if/else
    statements from run.run

    Args:
        input_data (tuple):  x_train, y_train, x_test, y_test
        **kwargs (iterable): Optional arguments to pass through.

    Returns:
        (tuple): input_data unchanged.
        or
        (tuple): input_data unchanged, feature_names if given in kwargs.
    """
    if 'feature_names' in kwargs:
        output = (input_data, kwargs['feature_names'])
    else:
        output = input_data
    return output

def make_unique(input_array):
    """
    Transforms input array so that each row of the array has the same elements
    as every other row, just in different orders. Removes any element that Does
    not appear in every row of the input array.

    Args:
        input_array list(list): The array to transform.

    Returns:
        None
    """
    remove_from_row = lambda x,row:row.remove(x) if x in row else None
    remove_from_array = lambda x,arr:map(lambda y:remove_from_row(x,y), arr)

    check = lambda x,row:True if x in row else False
    check_all_rows = lambda x,arr:True if np.asarray(map(lambda y:check(x,y),arr)).all() else False

    remove_element = lambda x,arr: remove_from_array(x,arr) if not check_all_rows(x,arr) else None

    remove_all = lambda arr: map(lambda x: remove_element(x,arr), [elem for subarr in arr for elem in subarr])

    remove_all(input_array)


def combine_lists(input_lists):
    """
    Performs a rank aggregation on a set of input lists. The ranking of element x
    is determined by the sum (1/(l*i)) for each list where l is the number of
    lists being aggregated and i is the position of x in the current list. The
    position of the first element in a list is considered to be 1 for the
    purpose of this method

    Args:
        input_lists (list(list)): A list of lists to be aggregated together.

    Returns:
        list: A ranked (from best to worst) list of all unique elements in
              input_lists.
    """

    length_of_list = len(input_lists[0])
    num_of_lists = len(input_lists)
    weight = 1.0/(length_of_list*num_of_lists)

    all_features = [elem for ranked_list in input_lists for elem in ranked_list]
    unique_features = np.unique(np.asarray(all_features)).tolist()
    feature_rankings = {k:0 for k in unique_features}

    for ranked_list in input_lists:
        ranked_list = list(ranked_list)
        for elem in ranked_list:
            val = weight*(length_of_list - ranked_list.index(elem))
            feature_rankings[elem] += val
    output = sorted(feature_rankings, key=lambda k: feature_rankings[k], reverse=True)
    output = [(x,feature_rankings[x]) for x in output]
    return output
