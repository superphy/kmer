import csv
import os
import random
import json
import pandas as pd
import numpy as np

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
            bad_files.append(temp.read().split('\n'))
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


def shuffle(A, B, labelA, labelB):
    """
    Returns "data", a combined and shuffled version of datasets A and B, as well
    as a secondary list that labels each element in "data" as originally
    belonging to A or B
    """
    if type(A) == list:
        data = A + B
        labels = [labelA for x in A] +  [labelB for x in B]
        return same_shuffle(data, labels)
    elif type(A) == np.ndarray:
        data = np.concatenate((A,B), axis=0)
        labelsA = np.full(A.shape[0], labelA)
        labelsB = np.full(B.shape[0], labelB)
        labels = np.concatenate((labelsA, labelsB), axis=0)
        data, labels = same_shuffle(data, labels)
        return np.asarray(data), np.asarray(labels)

def flatten(data):
    """
    Takes a 3D numpy ndarray and makes it 2D
    """
    data = data.reshape(data.shape[0], data.shape[1])
    return data

def make3D(data):
    """
    Takes a 2D numpy ndarray and makes it 3D
    """
    data = data.reshape(data.shape + (1,))
    return data


def sensitivity_specificity(predicted_values, true_values):
    """
    Takes two arrays, one is the predicted_values from running a prediction, the
    other is the true values. Returns the sensitivity and the specificity of the
    machin learning model.
    """
    true_pos = len([x for x in true_values if x == 1])
    true_neg = len([x for x in true_values if x == 0])
    false_pos = 0
    false_neg = 0
    for i in range(len(predicted_values)):
        if true_values[i] == 0 and predicted_values[i] == 1:
            false_pos += 1
        if true_values[i] == 1 and predicted_values[i] == 0:
            false_neg += 1

    sensitivity = (1.0*true_pos)/(true_pos + false_neg)
    specificity = (1.0*true_neg)/(true_neg + false_pos)

    return sensitivity, specificity


def parse_metadata(metadata, pos_label, neg_label, pos_path='', neg_path='',
                   train_label='', test_label='', file_suffix='', sep='\t'):
    """
    Parameters:
        metadata:    A csv metadata sheet with 2 or 3 columns. The first row
                     should be column headers.
                     Col 1: The name of the genome
                     Col 2: The classification label for the genome.
                     Col 3: If present: Whether or not the genome is for
                            training or testing.
                            If not present: 80%% of genomes are placed in
                            training, the rest in testing.
        pos_label:   Label used to classify positive genomes in "metadata".
        neg_label:   Label used to classify negative genomes in "metadata".
        pos_path:    The path to the fasta files for positive genomes.
        neg_path:    The path to the fasta files for negative genomes.
        train_label: Label identifying train genomes in "metadata".
        test_label:  Label identifying test genomes in "metadata".
        file_suffix: Suffix to appened to genome names in "metadata".
        sep:         The delimiter used in "metadata"

    Returns:
        x_train:   All the training fasta files
        y_train:   The labels for x_train
        x_test:    All the test fasta files.
        y_test:    The labels for x_test
    """
    with open(metadata, 'r') as f:
        pos_train = []
        neg_train =[]
        pos_test = []
        neg_test = []
        headers = f.readline()
        line = f.readline()
        while line:
            line = line.rstrip('\n')
            line = line.split(sep)
            if train_label:
                if line[1] == pos_label and line[2] == train_label:
                    pos_train.append(pos_path+line[0]+file_suffix)
                elif line[1] == neg_label and line[2] == train_label:
                    neg_train.append(neg_path+line[0]+file_suffix)
                elif line[1] == pos_label and line[2] == test_label:
                    pos_test.append(pos_path+line[0]+file_suffix)
                elif line[1] == neg_label and line[2] == test_label:
                    neg_test.append(neg_path+line[0]+file_suffix)
            else:
                if line[1] == pos_label:
                    pos_train.append(pos_path+line[0]+file_suffix)
                elif line[1] == neg_label:
                    neg_train.append(neg_path+line[0]+file_suffix)
            line = f.readline()
        if not train_label:
            random.shuffle(pos_train)
            random.shuffle(neg_train)
            cutoff = int(0.8*len(pos_train))
            pos_test = pos_train[cutoff:]
            pos_train = pos_train[:cutoff]
            cutoff = int(0.8*len(neg_train))
            neg_test = neg_train[cutoff:]
            neg_train = neg_train[:cutoff]

    x_train, y_train = shuffle(pos_train, neg_train, 1, 0)
    x_test, y_test = shuffle(pos_test, neg_test, 1, 0)

    return x_train, y_train, x_test, y_test


def parse_json(path='/home/rboothman/moria/entero_db/', suffix='.fasta',
               key='assembly_barcode', *json_files):
    """
    Parameters:
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

def parse_salmonella_metadata(metadata='/home/rboothman/PHAC/kmer/Data/amr_sorted.csv',
                              antibiotic='ampicillin',
                              path='/home/rboothman/Data/salmonella_amr/',
                              suffix='.fna', pos_label='Susceptible',
                              neg_label='Resistant'):
    data = pd.read_csv(metadata)
    filtered = data[data.Antibiotic == antibiotic]
    susceptible = filtered[filtered.AMR == pos_label]
    resistant = filtered[filtered.AMR == neg_label]
    all_files = os.listdir(path)
    susceptible = [path+str(x)+suffix for x in list(susceptible.Fasta) if str(x)+suffix in all_files]
    resistant = [path+str(x)+suffix for x in list(resistant.Fasta) if str(x)+suffix in all_files]
    x_train, y_train = shuffle(susceptible, resistant, 1, 0)

    cutoff = int(0.8*len(x_train))
    x_test = x_train[cutoff:]
    y_test = y_train[cutoff:]

    x_train = x_train[:cutoff]
    y_train = y_train[:cutoff]

    return (x_train, y_train, x_test, y_test)
