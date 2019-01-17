#!/usr/bin/env python


import numpy as np
from numpy.random import seed
import pandas as pd
from pandas import DataFrame
import sys
import pickle
from decimal import Decimal
import os
from sklearn import preprocessing

import tensorflow
from tensorflow import set_random_seed

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report, precision_recall_fscore_support

from model_evaluators import *
from data_transformers import *

"""
Comments in the data and create_model functions can cause unintended behaviour so all comments will be here
Save x_train, y_train, x_test, y_test right after line 'x_test  = sk_obj.transform(x_test)' of feature selection
The script will try up to 5 hidden layers, reporting number of layers, number of neurons in each hidden layer,
and dropout between each of the layers.
Output to std.out contains # of layers, # of neurons, dropout rates, EarlyStopping & ReduceLROnPlateau patience
as well as the accuracy of the optimized model 
"""

def data():
	from keras.utils import to_categorical
	num_classes = 9
	x_train = np.load('x_train.npy')
	y_train = np.load('y_train.npy')
	x_test = np.load('x_test.npy')
	y_test = np.load('y_test.npy')
	new_val = 1
	y_train = to_categorical(y_train, num_classes)
	y_test  = to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
	patience = {{choice([4,8,12,16])}}
	early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
	model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
	reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

	model = Sequential()

	model.add(Dense(x_train.shape[1],activation='relu',input_dim=(x_train.shape[1])))
	model.add(Dropout({{uniform(0,1)}}))

	num_layers = {{choice(['zero', 'one', 'two', 'three', 'four', 'five'])}}

	if ((num_layers == 'one') or (num_layers == 'two') or (num_layers == 'three') or (num_layers == 'four') or (num_layers == 'five')):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if ((num_layers == 'two') or (num_layers == 'three') or (num_layers == 'four') or (num_layers == 'five')):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if ((num_layers == 'three') or (num_layers == 'four') or (num_layers == 'five')):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if ((num_layers == 'four') or (num_layers == 'five')):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))
	if (num_layers == 'five'):
		model.add(Dense(int({{uniform(num_classes,x_train.shape[1])}})))
		model.add(Dropout({{uniform(0,1)}}))

	model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

	model.compile(loss='poisson', metrics=['accuracy'], optimizer='adam')
	model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[early_stop, reduce_LR])

	score, acc = model.evaluate(x_test, y_test, verbose=0)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == "__main__":

	train_data, train_names, test_data, test_names = data()
	best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=100, trials=Trials())
	print(best_model.evaluate(test_data, test_names))
	print("Parameters of best run", best_run)
	results = ann_1d(best_model, test_data, test_names, 0)
	labels = np.arange(0,9)
	avg_reports = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)
	avg_reports = np.transpose(avg_reports)
	avg_reports = np.around(avg_reports, decimals=2)
	print(avg_reports)
