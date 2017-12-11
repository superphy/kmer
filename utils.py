import csv
import os
import random
import json
import sys
import pandas as pd
import numpy as np
import constants
from sklearn.preprocessing import LabelEncoder

def setup_files(filepath):
    """
    Takes a path to a directory, returns a list of the complete paths to each
    file in the directory
    """
    if not filepath[-1] == '/':
        filepath += '/'
    return [filepath + x for x in os.listdir(filepath)]


def check_fasta(file):
    """
    Returns True if the file is fasta or fastq, False otherwise.
    """
    with open(file, 'r') as f:
        firstline = f.readline()
        if firstline[0] == '>' or firstline[0] == '@':
            return True
        else:
            return False


def valid_file(test_files, *invalid_files):
    """
    Params:
        test_files:     a list of fasta file names.
        *invalid_files: one or more files that contain a list of invalid fasta
                        file names.
    Returns:
        test_files with any files that appeared in *invalid files removed.
    """
    bad_files = []
    for file in invalid_files:
        with open(file, 'r') as temp:
            bad_files.extend(temp.read().split('\n'))
    return [x for x in test_files if x not in bad_files]


def same_shuffle(a,b):
    """
    Shuffles two lists so that the elements at index x in both lists before
    shuffling are at index y in their respective list after shuffling.
    """
    temp = list(zip(a, b))
    random.shuffle(temp)
    a,b = zip(*temp)
    return list(a),list(b)


def shuffle(data, labels):
    try:
        assert(len(data) == len(labels))
        assert(type(data[0]) == list or type(data[0]) == np.ndarray)
    except AssertionError as E:
        print E
    if type(data[0]) == list:
        all_data = []
        for x in data: all_data.extend(x)
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
    Takes a 3D numpy ndarray and makes it 2D
    """
    data = data.reshape(data.shape[0], data.shape[1])
    return data

def make3D(data):
    """
    Takes a 2D numpy ndarray and makes it 3D to be used in a Conv1D keras layer.
    """
    data = data.reshape(data.shape[0], data.shape[1], 1)
    return data

def sensitivity_specificity(predicted_values, true_values):
    """
    Args:
        predicted_values:   Array, what the model predicted.
        true_values:        Array, the true values.
    Returns:
        results:    Dictionary, keys are the classes, values are dictionaries
                    with keys 'sensitivity' and 'specificity'
    """
    results = {}
    if type(predicted_values) != np.ndarray:
        predicted_values = np.asarray(predicted_values)
    if type(true_values) != np.ndarray:
        true_values = np.asarray(true_values)

    classes = np.unique(true_values)
    for c in classes:
        predicted_pos = np.where(predicted_values==c, 1, 0)
        predicted_neg = np.where(predicted_values!=c, 1, 0)
        true_pos = np.where(true_values==c, 1, 0)
        true_neg = np.where(true_values!=c, 1, 0)

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
                   test_label='Test', suffix='', prefix='',sep=None,
                   one_vs_all=None, remove=None, validate=True, blacklist=None):
    """
    Args:
        metadata:     A csv file, must contain at least one column of genome
                      names and one column of their classifications.
        fasta_header: String, header for the genome name column
        label_header: String, header for the genome classification column
        train_header: String, header for the column that contains train/test
                      labels, if not given a random 80/20 split will be used to
                      generate the train/test datasets.
        train_label:  String, labels train genomes under train_header.
        test_label:   String, labels test genomes under train_header.
        extra_header: String, header for an additional column.
        extra_label:  String, if a sample's value under the extra_header column
                      does not match extra_label it will be removed.
        prefix:       String, prefix to attach to the front of genome
                      names, for instance the complete filepath.
        suffix:       String, suffix to appened to the genome names for instance
                      .fasta
        sep:          The delimiter used in metadata, if None the
                      delimiter is guessed.
        one_vs_all:   String, changes a multiclass problem into a binary
                      problem. All samples whose classification does not match
                      one_vs_all will be combined into one class.
    Returns:
        x_train:   All the training fasta files
        y_train:   The labels for x_train
        x_test:    All the test fasta files.
        y_test:    The labels for x_test
    """
    if sep is None:
        data = pd.read_csv(metadata, sep=sep, engine='python')
    else:
        data = pd.read_csv(metadata, sep=sep)
    if extra_header:
        data = data[data[extra_header] == extra_label]
    if remove:
        data = data.drop(data[data[label_header]==remove].index)
    if blacklist is not None:
        data = data.drop(data[data[fasta_header].isin(blacklist)].index)
    if one_vs_all:
        data[label_header] = data[label_header].where(data[label_header]==one_vs_all, 'Other')
    all_labels = np.unique(data[label_header])
    all_labels = all_labels[~pd.isnull(all_labels)]
    if train_header:
        train_data = data[data[train_header]==train_label]
        test_data = data[data[train_header]==test_label]
        all_train_data = []
        all_test_data = []
        for label in all_labels:
            all_train_data.append(train_data[train_data[label_header]==label])
            if validate:
                all_test_data.append(test_data[test_data[label_header]==label])
        all_train_data = [x[fasta_header].values for x in all_train_data]
        if validate:
            all_test_data = [x[fasta_header].values for x in all_test_data]
        else:
            all_test_data = test_data[fasta_header].values
    else:
        all_train_data = []
        all_test_data = []
        for label in all_labels:
            label_data = data[data[label_header]==label]
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
        y_test = np.array([])

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
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            fasta_names = [str(x[key]) for x in data]
            fasta_names = [x for x in fasta_names if valid_file(x)]
            fasta_names = [path + x + suffix for x in fasta_names]
            fasta_names = [x for x in fasta_names if check_fasta(x)]
        output.append(fasta_names)

    return output


def convert_to_numerical_classes(data):
    """
    Uses a scikitlearn LabelEncoder to convert y_train and y_test (if it exists)
    to numerical labels, returns data as well as the label encoder to allow the
    labels to be converted back.
    """
    le = LabelEncoder()
    if len(data) > 3:
        labels = data[1]+data[3]
        le.fit(labels)
        output_data = (data[0],le.transform(data[1]),data[2],le.transform(data[3]))
    else:
        le.fit(data[1])
        output_data = (data[0], le.transform(data[1]), data[2])
    return output_data, le
