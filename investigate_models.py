from keras.models import load_model
# from sklearn.model_selection import StratifiedShuffleSplit as SSS
# from ecoli_human_bovine import setup_files
# from kmer_counter import get_counts, count_new_kmers
# from sklearn.preprocessing import MinMaxScaler
# from verify_results import setup
# from get_fasta_from_json import get_fasta_from_json
# import os
# from find_params import data_us_uk
# import random
# from test import get_vals
# from verify_results import data_us_uk
from keras.models import load_model, model_from_json
from kmer_counter import get_counts, count_kmers, add_counts
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, model_from_json
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Conv1D
import random
import os
from get_fasta_from_json import train_test
from US_UK_data import get_filepaths, get_preprocessed_data
from salmonella_amr import setup_files
from presence_absence_data import get_data_and_labels
from feature_selection import variance_threshold, select_percentile, select_k_best
from data_augmentation import augment_data_adasyn, augment_data_smote, augment_data_naive

# enterobase = '/home/rboothman/moria/enterobase_db/'
#
# moria = [enterobase + x for x in os.listdir(enterobase)]
#
# model98a = load_model('Models/ModelB2017-10-18-13-45-0.981818177483.h5')
#model = load_model('Models/ModelB2017-10-18-13-46-0.963636365804.h5')
# model98b = load_model('Models/ModelB2017-10-18-13-46-0.981818182902.h5')
# model = load_model('Models/ModelB2017-10-18-13-47-1.0.h5')
# model = load_model('MODEL.h5')
# model100c = load_model('Models/ModelB2017-10-18-13-49-1.0.h5')
# model100d = load_model('Models/ModelB2017-10-18-13-50-1.0.h5')
# model100e = load_model('Models/ModelB2017-10-18-13-51-1.0.h5')
# model100f = load_model('Models/ModelB2017-10-18-13-52-1.0.h5')
#
# models = [model100a, model100b, model100c, model100d, model100e, model100f]
# # layers = []
# #
# # for model in models:
# #     models_weights = []
# #     for layer in model.layers:
# #         weights = leyer.get_weights()
# #         models_weights.append(weights)
# #     layers.append(models_weights)
# #
# # for i in range(len(layers[0][0])):
# #     layers[0][i]
# zero_zero = []
# zero_one = []
# three_zero = []
# three_one = []
#
# for model in models:
#     first_layer = model.layers[0]
#     last_layer = model.layers[3]
#     zero_zero.append(first_layer.get_weights()[0])
#     zero_one.append(first_layer.get_weights()[1])
#     three_zero.append(last_layer.get_weights()[0])
#     three_one.append(last_layer.get_weights()[1])
#
# a = np.mean(zero_zero, axis=0)
# b = np.mean(zero_one, axis=0)
# c = np.mean(three_zero, axis=0)
# d = np.mean(three_one, axis=0)
#
# print zero_zero[0]
# print zero_zero[1]
# print a
# garbage, garbage1, X, labels = data_us_uk()
# human, bovine = get_fasta_from_json()
# human = [x for x in human if x in moria]
# bovine =[x for x in bovine if x in moria]
# fasta_files = human + bovine
# labels = [1 for x in human] + [0 for x in bovine]
# temp = list(zip(fasta_files, labels))
# random.shuffle(temp)
# fasta_files, labels = zip(*temp)
# fasta_files = list(fasta_files)
# labels = list(labels)
#
# # count_new_kmers(fasta_files, "database2")
# X = get_counts(fasta_files, "database2")
# scaler = MinMaxScaler(feature_range=(-1,1))
# X = scaler.fit_transform(X)
#
# X = X.reshape(X.shape + (1,))
#
# scores = model.evaluate(X, labels, batch_size=10, verbose=1)
# print scores
#
# s = SSS(n_splits=10, test_size = 0.1, random_state = 13)
# for indices in s.split(genomes, labels):
#     X_test = [genomes[x] for x in indices[0]]
#     Y_test = [labels[x] for x in indices[0]]
#     scaler = MinMaxScaler(feature_range=(-1,1))
#     X_test = scaler.fit_transform(X_test)
#     X_test = X_test.reshape(X_test.shape + (1,))
#
#     scores = model100a.evaluate(X_test, Y_test, batch_size=10, verbose=1)
#
#     print "\n", scores

# def get_files():
#     sus, res = setup_files('kanamycin')
#     labels = [1 for x in sus] + [0 for x in res]
#     all_files = sus+res
#
#     temp = list(zip(all_files, labels))
#     random.shuffle(temp)
#     all_files, labels = zip(*temp)
#     all_files = list(all_files)
#     labels = list(labels)
#
#     cutoff = int(0.8*len(all_files))
#
#     x_train = all_files[:cutoff]
#     y_train = labels[:cutoff]
#     x_test = all_files[cutoff:]
#     y_test = labels[cutoff:]
#     return x_train, y_train, x_test, y_test
#     # return data("database", True, False, 1, 1)
#
# #database: ecoli_human_bovine
# #database2: salmonella amr
#
# def get_vals():
#     x_train, y_train, x_test, y_test = train_test('human_metadata', 'bovine_metadata')
#
#     return x_train+x_test, y_train+y_test


# uk_data, uk_labels, us_data, us_labels = data("database", True, False, 1, 1)
#
# x_train, y_train = get_vals()
# x_train, y_train, x_test, y_test = get_augmented_data("database", True, False, 1, 1, 10)
x_train, y_train, x_test, y_test = get_preprocessed_data("database", True, False, 7, 13)
# x_train, y_train, x_test, y_test = get_data_and_labels('binary_table.txt', '/home/rboothman/Data/human_bovine/human/', True)
x_train, y_train, x_test, y_test = variance_threshold(x_train, y_train, x_test, y_test, 0.1)
x_train, y_train = augment_data_adasyn(x_train, y_train, 500)
# x = int(0.8*x_test.shape[0])

# x_test = x_train[x:]
# y_test = y_train[x:]
#
# x_train = x_train[:x]
# y_train = y_train[:x]
# x_train = x_test[x:]
# y_train = y_test[x:]
# x_test = x_test[:x]
# y_test = y_test[:x]
# temp = list(zip(x_train, y_train))
# random.shuffle(temp)
# x_train, y_train = zip(*temp)
# x_train = list(x_train)
# y_train = list(y_train)

# all_files = x_train + x_test
# count_kmers(7, 13, all_files, "database2")

# x_train = get_counts(x_train, "database")
# x_train = np.asarray(x_train, dtype='float64')
#
# x_test = get_counts(x_test, "database")
# x_test = np.asarray(x_test, dtype='float64')
#
# scaler = MinMaxScaler(feature_range=(-1,1))
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#
# x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.reshape(x_test.shape + (1,))

model = Sequential()
model.add(Conv1D(filters=10,
                 kernel_size=3,
                 activation='relu',
                 input_shape = (x_train.shape[1], 1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# model = Sequential()
# model.add(Conv1D(5, 7, activation='tanh', input_shape=(x_train.shape[1], 1)))
# model.add(AveragePooling1D(pool_size=7, strides=1))
# model.add(Dropout(0.5508))
# model.add(Conv1D(5, 7, activation='tanh'))
# model.add(AveragePooling1D(pool_size=7, strides=1))
# model.add(Dropout(0.5508))
# model.add(Conv1D(5, 7, activation='tanh'))
# model.add(AveragePooling1D(pool_size=7, strides=1))
# model.add(Dropout(0.5508))
# model.add(Conv1D(5, 7, activation='tanh'))
# model.add(AveragePooling1D(pool_size=7, strides=1))
# model.add(Dropout(0.5508))
# model.add(Conv1D(5, 7, activation='tanh'))
# model.add(AveragePooling1D(pool_size=7, strides=1))
# model.add(Dropout(0.5508))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# json_string = model.to_json()

# score = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
#
# new_model = model_from_json(json_string)

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=10, epochs=120, verbose=1)

print model.evaluate(x_test, y_test, batch_size=10, verbose=0)
# score1 = model.evaluate(uk_data, uk_labels, batch_size=10, verbose=0)
# score2 = model.evaluate(us_data, us_labels, batch_size=10, verbose=0)
# print score1[1], score2[1]
