#coding=utf-8

try:
    import numpy as np
except:
    pass

try:
    from numpy.random import seed
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    from pandas import DataFrame
except:
    pass

try:
    import sys
except:
    pass

try:
    import pickle
except:
    pass

try:
    from decimal import Decimal
except:
    pass

try:
    import os
except:
    pass

try:
    import tensorflow
except:
    pass

try:
    from tensorflow import set_random_seed
except:
    pass

try:
    from concurrent.futures import ProcessPoolExecutor
except:
    pass

try:
    from multiprocessing import cpu_count
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.layers.convolutional import Conv1D
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.layers import Flatten, BatchNormalization
except:
    pass

try:
    from keras.models import Sequential, load_model
except:
    pass

try:
    from keras.utils import np_utils, to_categorical
except:
    pass

try:
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from sklearn import metrics
except:
    pass

try:
    from sklearn.externals import joblib
except:
    pass

try:
    from sklearn.cross_validation import train_test_split
except:
    pass

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except:
    pass

try:
    from sklearn.feature_selection import SelectKBest, f_classif
except:
    pass

try:
    from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
except:
    pass

try:
    from model_evaluators import *
except:
    pass

try:
    from data_transformers import *
except:
    pass

try:
    from keras.utils import to_categorical
except:
    pass

try:
    from sklearn.feature_selection import SelectKBest, f_classif
except:
    pass

try:
    from collections import Counter
except:
    pass

try:
    from keras.utils import to_categorical
except:
    pass

try:
    from collections import Counter
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter

feats = int(sys.argv[1])
attribute  = sys.argv[2]
fold  = sys.argv[4]
dataset = sys.argv[5]
run = sys.argv[6]

# fold 1 uses sets 1,2,3 to train, 4 to test, fold 2 uses sets 2,3,4 to train, 5 to test, etc
train_sets = [(i+int(fold)-1)%5 for i in range(5)]
train_sets = [str(i+1) for i in train_sets]



# load the relevant training sets and labels
x_train1 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/x.npy'.format(dataset,attribute,train_sets[0]), allow_pickle = True)
x_train2 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/x.npy'.format(dataset,attribute,train_sets[1]), allow_pickle = True)
x_train3 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/x.npy'.format(dataset,attribute,train_sets[2]), allow_pickle = True)
y_train1 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/y.npy'.format(dataset,attribute,train_sets[0]), allow_pickle = True)
y_train2 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/y.npy'.format(dataset,attribute,train_sets[1]), allow_pickle = True)
y_train3 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/y.npy'.format(dataset,attribute,train_sets[2]), allow_pickle = True)
y_train4 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/y.npy'.format(dataset,attribute,train_sets[3]), allow_pickle = True)
y_train5 = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/y.npy'.format(dataset,attribute,train_sets[4]), allow_pickle = True)
all_y_trains = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))

# merge the 3 training sets into 1
x_train = np.vstack((x_train1, x_train2, x_train3))
y_train = np.concatenate((y_train1, y_train2, y_train3))

num_classes = max(Counter(all_y_trains).keys()) + 1

x_test  = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/x.npy'.format(dataset,attribute,train_sets[3]), allow_pickle = True)
y_test  = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/y.npy'.format(dataset,attribute,train_sets[3]), allow_pickle = True)

x_val  = np.load('data'+run+'/hyp_splits/{}-{}/splits/set{}/x.npy'.format(dataset,attribute,train_sets[4]), allow_pickle = True)

# hyperas asks for train and test so the validation set is what comes last, to check the final model
# we need to save it to be used later, because we have the sk_obj now.
if(feats!=0):
	sk_obj = SelectKBest(f_classif, k=feats)
	x_train = sk_obj.fit_transform(x_train, y_train)
	x_test  = sk_obj.transform(x_test)
	x_val  = sk_obj.transform(x_val)
	np.save('data'+run+'/hyp_splits/{}-{}/splits/val{}_{}.npy'.format(dataset,attribute,fold,str(feats)), x_val)

y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

#print(y_test)


def keras_fmin_fnct(space):

	patience = space['patience']
	early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
	model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
	reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

	model = Sequential()
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, num_classes)
	#print(y_test)
	# how many hidden layers are in our model
	num_layers = space['num_layers']

	if(num_layers == 'zero'):
		model.add(Dense(num_classes,activation='softmax',input_dim=x_train.shape[1]))
	else:
		# this isnt a for loop because each variable needs its own name to be independently trained
		if (num_layers in ['one','two','three','four','five']):
			model.add(Dense(int(space['int']),activation='relu',input_dim=x_train.shape[1]))
			model.add(Dropout(space['Dropout']))
		if (num_layers in ['two','three','four','five']):
			model.add(Dense(int(space['int_1'])))
			model.add(Dropout(space['Dropout_1']))
		if (num_layers in ['three','four','five']):
			model.add(Dense(int(space['int_2'])))
			model.add(Dropout(space['Dropout_2']))
		if (num_layers in ['four','five']):
			model.add(Dense(int(space['int_3'])))
			model.add(Dropout(space['Dropout_3']))
		if (num_layers == 'five'):
			model.add(Dense(int(space['int_4'])))
			model.add(Dropout(space['Dropout_4']))

		model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

	model.compile(loss='poisson', metrics=['accuracy'], optimizer='adam')
	model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[early_stop, reduce_LR])

	score, acc = model.evaluate(x_test, y_test, verbose=0)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'patience': hp.choice('patience', [4,8,12,16]),
        'num_layers': hp.choice('num_layers', ['zero', 'one', 'two', 'three', 'four', 'five']),
        'int': hp.uniform('int', num_classes,x_train.shape[1]),
        'Dropout': hp.uniform('Dropout', 0,1),
        'int_1': hp.uniform('int_1', num_classes,x_train.shape[1]),
        'Dropout_1': hp.uniform('Dropout_1', 0,1),
        'int_2': hp.uniform('int_2', num_classes,x_train.shape[1]),
        'Dropout_2': hp.uniform('Dropout_2', 0,1),
        'int_3': hp.uniform('int_3', num_classes,x_train.shape[1]),
        'Dropout_3': hp.uniform('Dropout_3', 0,1),
        'int_4': hp.uniform('int_4', num_classes,x_train.shape[1]),
        'Dropout_4': hp.uniform('Dropout_4', 0,1),
    }
