import csv
import os
import random
import json


def get_human_path():
    """
    Returns the file path to the directory containing all the human fasta files.
    """
    return '/home/rboothman/Data/human_bovine/human/'


def get_bovine_path():
    """
    Returns the file path to the directory containing all the bovine fasta files
    """
    return '/home/rboothman/Data/human_bovine/bovine/'


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


def parse_genome_region_table(table, *params):
    """
    Parameters:
        table:      A binary_table.txt output by panseq.
        *params:    Parameters necessary to run parse_metadata.

    Returns:
        pos_train_data: The binary data for all the positive training genomes.
        neg_train_data: The binary data for all the negative training genomes.
        pos_test_data:  THe binary data for all the positive test genomes.
        neg_test_data:  The binary data for all the negative test genomes.
    """
    labels = parse_metadata(*params)
    x_train_labels = labels[0]
    y_train = labels[1]
    x_test_labels = labels[2]
    y_test = labels[3]

    x_train = []
    x_test = []

    for header in x_train_labels:
        output = []
        with open(table, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                output.append(row[header])
        x_train.append(output)

    for header in x_test_labels:
        output = []
        with open(table, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                output.append(row[header])
        x_test.append(output)

    return x_train, y_train, x_test, y_test


def parse_metadata(metadata, pos_label, neg_label, pos_path='', neg_path='',
                   train_label='', test_label='', file_suffix=''):
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

    Returns:
        pos_train:   All postive training fasta files.
        neg_train:   All negative training fasta files.
        pos_test:    All positive test fasta files.
        neg_test:    All negative test fasta files.
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
            line = line.split(',')
            if train_label:
                if line[1] == pos_label and line[2] == train_label:
                    pos_train.append(pos_path+line[0]+file_suffix)
                elif line[1] == neg_label and line[2] == train_label:
                    neg_train.append(neg_path+line[0]+file_suffix)
                elif line[1] == pos_label and line[2] == test_label:
                    pos_test.append(pos_path+line[0]+file_suffix)
                elif line[1] == neg_path and line[2] == test_label:
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
            fasta_names = [path+x+suffix for x in fasta_names]
            fasta_names = [x for x in fasta_names if check_fasta(x)]
        output.append(fasta_names)

    return output
