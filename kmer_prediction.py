from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers.convolutional import Conv1D, Conv2D

from keras.layers import Dense, Activation, Dropout, RepeatVector, LSTM
from keras.layers import Flatten, Reshape, Input, concatenate, BatchNormalization
from keras.layers import LocallyConnected1D, GaussianNoise
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model

from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.preprocessing import MinMaxScaler

from kmer_counter import count_kmers, get_counts

import numpy as np
import os
import sys


def set_up_files(filepath):
    """
    Helper method for running kmer_prediction.py from the command line.

    Takes a path to a directory, returns a list of the complete paths to each
    file in the directory
    """
    if not filepath[-1] == '/':
        filepath += '/'
    return [filepath + x for x in os.listdir(filepath)]


def sensitivity_specificity(predicted_values, true_values):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Takes two arrays, one is the predicted_values from running a prediction, the other is
    the true values. Returns the sensitivity and the specificity of the machine
    learning model.
    """
    true_pos = len([x for x in true_values if x == 1])
    true_neg = len([x for x in true_values if x == 0])
    false_pos = 0
    false_neg = 0
    err_rate = 0
    for i in range(len(predicted_values)):
        if true_values[i] == 0 and predicted_values[i] == 1:
            false_pos += 1
            err_rate += 1
        if true_values[i] == 1 and predicted_values[i] == 0:
            false_neg += 1
            err_rate += 1

    sensitivity = (1.0*true_pos)/(true_pos + false_neg)
    specificity = (1.0*true_neg)/(true_neg + false_pos)
    score = len(predicted_values - 1.0*err_rate)/len(predicted_values)

    return score, sensitivity, specificity


def aNeuralNet(length):
    model = Sequential()
    model.add(Dense(32, input_dim = length))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = 'sgd',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    return model


def bNeuralNet(length):
    model = Sequential()
    model.add(Conv1D(15, 3, activation='relu', input_shape=(length,1)))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    return model


def cNeuralNet(length):
    model = Sequential()
    model.add(Conv1D(15, 5, activation='relu', input_shape=(length,1)))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    return model

def dNeuralNet(length):
    model = Sequential()
    model.add(Dense(128, input_dim=length))
    model.add(Activation(None))
    model.add(Dropout(0.5990))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.1643))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = 'sgd',
                    loss = 'binary_crossentropy',
                    metrics=['accuracy'])
    return model

def eNeuralNet(length):
    model = Sequential()
    model.add(Dense(20, activation=None, input_dim=length))
    model.add(Dropout(0.5523))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5940))
    model.add(Dense(10, activation=None))
    model.add(Dropout(0.2694))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def make_predictions(train_data, train_labels, test_data, test_labels):
    train_data = np.asarray(train_data, dtype='float64')
    train_labels = np.asarray(train_labels)
    train_labels = train_labels.reshape(train_labels.shape[0], 1)

    test_data = np.asarray(test_data, dtype='float64')

    modelA = aNeuralNet(train_data.shape[1])
    modelB = bNeuralNet(train_data.shape[1])
    modelC = cNeuralNet(train_data.shape[1])
    modelD = dNeuralNet(train_data.shape[1])

    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(train_data)
    Z = scaler.transform(test_data)

    Xprime = X.reshape(X.shape + (1,))
    Zprime = Z.reshape(Z.shape + (1,))

    try:
        modelA.fit(X, train_labels, epochs=25, batch_size=10, verbose=1)
        modelB.fit(Xprime, train_labels, epochs=25, batch_size=10, verbose=1)
        # modelC.fit(Xprime, train_labels, epochs=25, batch_size=10, verbose=1)
        modelD.fit(X, train_labels, epochs=25, batch_size=32, verbose=1)

        if test_labels:
            test_labels = np.asarray(test_labels)
            test_labels = test_labels.reshape(test_labels.shape[0], 1)

            scoreA = modelA.evaluate(Z, test_labels, batch_size=32, verbose=1)
            scoreB = modelB.evaluate(Zprime, test_labels, batch_size=10, verbose=1)
            # scoreC = modelC.evaluate(Zprime, test_labels, batch_size=10, verbose=1)
            scoreD = modelD.evaluate(Z, test_labels, batch_size=10, verbose=1)
            return (scoreA[1], scoreB[1], scoreD[1])
        else:
            a = modelA.predict(Z)
            b = modelB.predict(Zprime)
            c = modelC.predict(Zprime)
            return (a, b, c)
    except (ValueError, TypeError) as E:
        print E
        return (-1, -1, -1)


def run(k, limit, num_splits, pos, neg, predict):
    if not predict:
        files = pos + neg
    else:
        files = pos + neg + predict

    #count_kmers( k, limit, files "database")

    arrays = get_counts(files, "database")

    labels=[1 for x in pos]+[0 for x in neg]

    if not predict:
        sss = SSS(n_splits=num_splits, test_size=0.2, random_state=42)

        scoreA_total = 0.0
        scoreB_total = 0.0
        scoreC_total = 0.0
        for indices in sss.split(arrays, labels):
            X = [arrays[x] for x in indices[0]]
            Y = [labels[x] for x in indices[0]]
            Z = [arrays[x] for x in indices[1]]
            ZPrime = [labels[x] for x in indices[1]]

            scores = make_predictions(X,Y,Z,ZPrime)
            print scores
            scoreA_total += scores[0]
            scoreB_total += scores[1]
            scoreC_total += scores[2]

        output = (scoreA_total/num_splits, scoreB_total/num_splits,
                    scoreC_total/num_splits)

    else:
        # sss = SSS(n_splits=1, test_size = 0.5, random_state=13)
        #
        # for indices in sss.split(arrays[:(len(pos)+len(neg))], labels):
        #     X = [arrays[x] for x in indices[0]]
        #     X.extend([arrays[x] for x in indices[1]])
        #     Y = [labels[x] for x in indices[0]]
        #     Y.extend([labels[x] for x in indices[1]])
        X = arrays[:len(pos) + len(neg)]
        Z = arrays[len(pos) + len(neg):]
        output = make_predictions(X, labels, Z, None)

    return output
