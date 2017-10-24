from kmer_counter import count_kmers, get_counts
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random



def same_shuffle(a,b):
    """
    Takes two lists and shuffles them so that the elements that appear in
    position x in both lists before the shuffling, both appear in position y
    after the shuffling.
    Returns the shuffled versions of the lists.
    """
    temp = list(zip(a,b))
    random.shuffle(temp)
    a,b = zip(*temp)
    return list(a), list(b)



def get_filepaths():
    """
    Parameters: None.
    Returns:
        x_train:    Shuffled list of filepaths to genomes from the UK dataset.
        y_train:    Labels corresponding to x_train.
        x_test:     Shuffled list of filepaths to genomes from the US dataset.
        y_test:     Labels corresponding to x_test.

    This method returns the filepaths to the fasta files to recreate the results
    of the 2016 paper "Support vector machine applied to predict the zoonotic
    potential of E. coli O157 cattle isolates" by Lupolova et. al.
    """
    human_path = '/home/rboothman/Data/human_bovine/human/'
    bovine_path = '/home/rboothman/Data/human_bovine/bovine/'

    with open('human_bovine.csv', 'r') as f:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        line = f.readline()
        while line:
            line = line.rstrip('\n')
            line = line.split(',')
            if line[1] == 'Human' and line[2] == 'Train':
                x_train.append(human_path+line[0]+'.fasta')
                y_train.append(1)
            elif line[1] == 'Bovine' and line[2] == 'Train':
                x_train.append(bovine_path+line[0]+'.fasta')
                y_train.append(0)
            elif line[1] == 'Human' and line[2] == 'Test':
                x_test.append(human_path+line[0]+'.fasta')
                y_test.append(1)
            elif line[1] == 'Bovine' and line[2] == 'Test':
                x_test.append(bovine_path+line[0]+'.fasta')
                y_test.append(0)
            line = f.readline()

    x_train, y_train = same_shuffle(x_train, y_train)
    x_test, y_test = same_shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test



def get_preprocessed_data(database, threeD, recount, k, l):
    """
    Parameters:
        database:   Path to an lmdb database where you want the kmer counts to
                    be stored.
        threeD:     Boolean value, if true the returned data has its dimensions
                    increased by 1. This is necessary if the data will be
                    passed to a neural network that uses convolutional layers.
        recount:    Boolean value, if true the kmers are recounted and stored in
                    the database, if false it is assumed that the database
                    already has the correct kmer counts and they are simply read
                    from the database.
        k:          Size of kmer to be counted. Ignored if recount is false.
        l:          Minimum number of times a kmer must appear in order to be
                    counted. Ignored if recount is false.
    Returns:
        x_train:    UK data ready to be input directly into a machine learning
                    model.
        y_train:    The human(1) bovine(0) labels corresponding to x_train.
        x_test:     US data ready to be input directly into a machine learning
                    model.
        y_test:     The human(1) bovine(0) labels corresponding to x_test.

    This method prepares the train and test data/labels to recreate the results
    of the 2016 paper "Support vector machine applied to predict the zoonotic
    potential of E. coli O157 cattle isolates" by Lupolova et. al.
    """
    x_train, y_train, x_test, y_test = get_filepaths()

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
