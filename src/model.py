#!/usr/bin/env python


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support
import collections
from sklearn.externals import joblib
import sys, os
from sklearn import preprocessing
import getopt
import pickle

from hpsklearn import HyperoptEstimator, svc, xgboost_classification
from hyperopt import tpe

from model_evaluators import *
from data_transformers import *


def get_data(train, predict_for):
	X = []
	Y = []
	if(train in ('kmer, omnilog')):
		X = np.load('data/filtered/'+predict_for+'/'+train+'_matrix.npy')
		Y = np.load('data/filtered/'+predict_for+'/'+train+'_rows_'+predict_for+'.npy')

	elif(train in ('uk', 'us','uk_us')):
		X = np.load('data/uk_us_unfiltered/kmer_matrix.npy')
		Y = np.load('data/uk_us_unfiltered/kmer_rows_Class.npy')

		if(train!='uk_us'):
			#the US dataset has been labeled as Test and the UK set as Train, we need to load the correct one
			dataset_array = np.load('data/uk_us_unfiltered/kmer_rows_Dataset.npy')
			if(train=='us'):
				us_mask = np.asarray([i =='Test' for i in dataset_array])
				X = X[us_mask]
				Y = Y[us_mask]
			else:
				uk_mask  = np.asarray([i =='Train'  for i in dataset_array])
				X = X[uk_mask]
				Y = Y[uk_mask]
	else:
		raise Exception('did not receive a valid -x or -y name, run model.py --help for more info')
	return X, Y

def usage():
	print("usage: model.py -x uk -y us -f 3000 -a Host\n\nOptions:")
	print(
	"-x, --train       Which set to train on [us, uk, uk_us, omnilog, kmer]",
	"-y, --test        Which set to test on [us, uk, uk_us, omnilog, kmer]",
	"                  Note that not passing a -y causes cross validation on the train set",
	"-f, --features    Number of features to train on, set to 0 to use all",
	"-a, --attribute   What to make the prediction on [Host, Serotype, Otype, Htype]",
	"-m, --model       Which model to use [XGB, SVM, ANN], defaults to XGB",
	"-o, --out         Where to save result DF, defaults to print to std out",
	"-p,               Add this flag to do hyperparameter optimization, XGB/SVM only",
	"-h, --help        Prints this menu",
	sep = '\n')
	return

if __name__ == "__main__":
	train = ''
	test = 'cv'
	num_feats = 0
	predict_for = ''
	model_type = 'XGB'
	hyper_param = 0
	out = 'print'

	try:
		opts, args =  getopt.getopt(sys.argv[1:],"hx:y:f:a:m:o:p",["help","train=","test=","features=","attribute=","model=","out="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg, in opts:
		if opt in ('-x', '--train'):
			train = arg
		elif opt in ('-y', '--test'):
			test = arg
		elif opt in ('-f', '--features'):
			num_feats = int(arg)
		elif opt in ('-a', '--attribute'):
			predict_for = arg
		elif opt in ('-m', '--model'):
			model_type = arg
		elif opt in ('-o', '--out'):
			out = arg
		elif opt == '-p':
			hyper_param = 1
		elif opt in ('-h', '--help'):
			usage()
			sys.exit()

	X = []
	Y = []
	x_train = []
	x_test = []
	y_train = []
	y_test = []

	le = preprocessing.LabelEncoder() # this encodes the classes into integers for the models (human, bovine, water -> 0,1,2)

	#if no -y is passed in we want to do a 5 fold cross validation on the data
	if test =='cv':
		X, Y = get_data(train, predict_for) # pull entire data set to be split later
		Y = le.fit_transform(Y)
		if(num_feats>= X.shape[1]):
			num_feats = 0
	else: #we are not cross validating
		x_train, y_train = get_data(train, predict_for) # pull training set
		x_test, y_test = get_data(test, predict_for) # pull testing set
		le.fit(np.concatenate((y_train,y_test))) # need to fit on both sets of labels, so everything is accounted for (finding what the replacements are)
		y_train = le.transform(y_train) # just applying the label encoder on the 2 sets (actually replacing things)
		y_test = le.transform(y_test)
		if(num_feats>= x_train.shape[1]):
			num_feats = 0

	# the omnilog crashes at 191, so 190 is our new 'all features'
	if((num_feats == 0 or num_feats>190) and train=='omnilog'):
		num_feats = 190

	num_classes = len(le.classes_)

	num_threads = 64

	cv = StratifiedKFold(n_splits=5, random_state=913824)
	cvscores = []
	window_scores = []
	mcc_scores = []
	report_scores = []
	split_counter = 0

	train_string = train
	test_string = test

	model_data = [[train, test]]
	if(test == 'cv'):
		model_data = cv.split(X,Y)

	"""
	If we didnt pass -y then we will be doing a 5 fold CV which means that this for loop will run 5 times, taking the average of each run
	If we did pass a -y then this will only run once
	"""
	for train,test in model_data:
		#split_counter +=1
		if(test_string=='cv'):
			x_train = X[train]
			x_test = X[test]
			y_test = Y[test]
			y_train = Y[train]

		#feature selection
		if(num_feats!=0):
			sk_obj = SelectKBest(f_classif, k=num_feats)
			x_train = sk_obj.fit_transform(x_train, y_train)
			x_test  = sk_obj.transform(x_test)
			"""
			#uncomment this section if you want to save datasets for hyp.property
			np.save('x_test.npy', x_test)
			np.save('x_train.npy', x_train)
			np.save('y_test.npy', y_test)
			np.save('y_train.npy', y_train)
			"""

		if(model_type == 'XGB'):
			if(num_classes==2 or (train_string == 'uk_us' and test_string == 'kmer')):
				objective = 'binary:logistic'
			else:
				objective = 'multi:softmax'
			if(hyper_param):
				model = HyperoptEstimator(classifier=xgboost_classification('xbc'), preprocessing=[], algo=tpe.suggest, trial_timeout=200)
			else:
				model = XGBClassifier(learning_rate=1, n_estimators=10, objective=objective, silent=True, nthread=num_threads)
			model.fit(x_train,y_train)
		elif(model_type == 'SVM'):
			from sklearn import svm
			if(hyper_param):
				model = HyperoptEstimator(classifier=svc("mySVC"), preprocessing=[], algo=tpe.suggest, trial_timeout=200)
			else:
				model = svm.SVC()
			model.fit(x_train,y_train)
		elif(model_type == 'ANN'):
			if(hyper_param):
				raise Exception('This script does not support hyperas for ANN hyperparameter optimization, see src/hyp.py')
			from keras.layers.core import Dense, Dropout, Activation
			from keras.models import Sequential
			from keras.utils import np_utils, to_categorical
			from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

			y_train = to_categorical(y_train, num_classes)
			y_test  = to_categorical(y_test, num_classes)

			patience = 16
			early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1, min_delta=0.005, mode='auto')
			model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
			reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 1, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

			model = Sequential()
			#This model currently has an input layer of num_feats neurons, 16% dropout, a hidden layer with 62 neurons, 44% dropout, and an output layer with num_classes outputs (or 1 if binary)
			model.add(Dense(num_feats,activation='relu',input_dim=(num_feats)))
			model.add(Dropout(0.16))
			model.add(Dense(62, activation='relu', kernel_initializer='uniform'))
			model.add(Dropout(0.44))

			if(num_classes==2 or (train_string == 'uk_us' and test_string == 'kmer')):
				loss = 'binary_crossentropy'
				num_outs = 1
			else:
				loss = 'poisson'
				num_outs = num_classes
			model.add(Dense(num_outs, kernel_initializer='uniform', activation='softmax'))
			model.compile(loss=loss, metrics=['accuracy'], optimizer='adam')

			model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[early_stop, reduce_LR])
		else:
			raise Exception('Unrecognized Model. Use XGB, SVM or ANN')

		if(model_type == 'ANN'):
			results = ann_1d(model, x_test, y_test, 0)
			#OBOResults = ann_1d(model, x_test, y_test, 1)
		else:
			results = xgb_tester(model, x_test, y_test, 0)
			#OBOResults = xgb_tester(model, x_test, y_test, 1)

		#window_scores.append(OBOResults[0])
		mcc_scores.append(results[1])

		labels = np.arange(0,num_classes)
		report = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)
		report_scores.append(report)
		cvscores.append(results[0])

	np.set_printoptions(suppress=True)
	avg_reports = np.mean(report_scores,axis=0)
	avg_reports = np.transpose(avg_reports)
	avg_reports = np.around(avg_reports, decimals=2)
	result_df = pd.DataFrame(data = avg_reports, index = le.classes_, columns = ['Precision','Recall', 'F-Score','Supports'])
	running_sum = 0
	t_string = ''
	if(test_string == 'cv'):
		result_df.values[:,3] = [i*5 for i in result_df.values[:,3]]
		t_string = 'aCrossValidation'
		for row in result_df.values:
			running_sum+=(row[1]*row[3]/X.shape[0])
	else:
		t_string = test_string
		for row in result_df.values:
			running_sum+=(row[1]*row[3]/(len(y_test)))
	if(train_string == 'uk_us'):
		train_string = 'ukus'
	if(t_string == 'uk_us'):
		t_string = 'ukus'
	print("Predicting for", predict_for)
	print("on {} features using a {} trained on {} data, tested on {}".format(num_feats, model_type, train_string, t_string))
	print("Accuracy:", running_sum)
	print(result_df)
	if(out!='print'):
		if not (out.endswith('/')):
			out = out + '/'
		out = out+predict_for+'_'+str(num_feats)+'feats_'+model_type+'trainedOn'+train_string+'_testedOn'+t_string+'.pkl'
		result_df.to_pickle(out)
