#!/usr/bin/env python

"""
This Script to split the training set into quarters so that we have
3/5th train, 1/5th test, 1/5th validate
"""

import os, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn import preprocessing

from model_evaluators import *
from data_transformers import *


if __name__=='__main__':
    """
    input 1 : dataset in [kmer, omnilog, uk, us, uk_us]
    input 2 : attribute in [Host, Serotype, Otype, Htype]
    """
    dataset = sys.argv[1]
    attribute = sys.argv[2]

    # note that the uk_us only has classification data for Host
    if (dataset in ['uk','us','uk_us'] and attribute != 'Host'):
        print("Skipping {} prediction in {} dataset".format(attribute, dataset))

    # for data that has been filtered
    if(dataset in ['kmer','omnilog']):
        # load the relevant data
        X = np.load("data/filtered/{}/{}_matrix.npy".format(attribute, dataset),allow_pickle=True)
        Y = np.load("data/filtered/{}/{}_rows_{}.npy".format(attribute, dataset, attribute),allow_pickle=True)
        Z = np.load("data/filtered/{}/{}_rows.npy".format(attribute, dataset),allow_pickle=True)

    # for unfiltered data of the uk_us set
    elif(dataset in ['uk','us','uk_us']):
        X = np.load("data/uk_us_unfiltered/kmer_matrix.npy",allow_pickle=True)
        Y = np.load("data/uk_us_unfiltered/kmer_rows_Class.npy",allow_pickle=True)
        Z = np.load("data/uk_us_unfiltered/kmer_rows.npy",allow_pickle=True)

        # us are labeled as train and uk are labeled as test, we need to return the correct ones only
        ukus_labels = np.load("data/uk_us_unfiltered/kmer_rows_Dataset.npy",allow_pickle=True)

        # create a bool mask to label rows matching what was passed in as dataset (sys.argv[1])
        if(dataset=='uk'):
            dataset_mask = [i=="Test" for i in ukus_labels]
        elif(dataset=='us'):
            dataset_mask = [i=="Train" for i in ukus_labels]
        else:
            dataset_mask = [True for i in ukus_labels]

        X = X[dataset_mask]
        Y = Y[dataset_mask]
        Z = Z[dataset_mask]


    else:
        raise Exception("Acceptable datasets are: [kmer, omnilog, uk, us, uk_us] but {} was given".format(dataset))


    # possible label encodings are determined possible label strings
    # for example, we change Bovine, Human, Human to 0,1,1

    le = preprocessing.LabelEncoder()

    # using fit with LabelEncoder isnt consistent across programs so we are going to manually set the attribute
    le.classes_ = list(set(Y))
    Y = le.transform(Y)

    # this can be changed to 6 if need be
    cv = StratifiedKFold(n_splits=5, random_state=913824)
    model_data = cv.split(X, Y, Z)

    set_count = 0
    for train, test in model_data:
        set_count+=1
        x_train = X[train]
        x_test  = X[test]
        y_test  = Y[test]
        y_train = Y[train]
        z_train = Z[train]
        z_test  = Z[test]

        # save data
        if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/hyp_splits/{}-{}/splits/set{}".format(dataset, attribute, set_count)):
            os.makedirs(os.path.abspath(os.path.curdir)+"/data/hyp_splits/{}-{}/splits/set{}".format(dataset, attribute, set_count), exist_ok = True)


        save_path = "data/hyp_splits/{}-{}/splits/set".format(dataset, attribute)+str(set_count)

        # This just saves the testing set, so the data is split into 5ths, each set is 1/5th of the data

        np.save(save_path+'/x.npy', x_test)
        np.save(save_path+'/y.npy', y_test)
        np.save(save_path+'/z.npy', z_test)
