from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers.convolutional import Conv1D, Conv2D

from keras.layers import Dense, Activation, Dropout, RepeatVector, LSTM
from keras.layers import Flatten, Reshape, Input, concatenate, BatchNormalization
from keras.layers import LocallyConnected1D, GaussianNoise
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model

import numpy as np
import inspect

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.optim import get_hyperopt_space, hyperparameter_names, get_hyperparameters
from hyperas.utils import remove_imports

from kmer_counter import count_kmers, get_counts

from ecoli_human_bovine import setup_files

from sklearn.model_selection import StratifiedShuffleSplit as SSS

from sklearn.preprocessing import MinMaxScaler


def data2D():
    pos, neg = setup_files()
    files = pos + neg
    arrays = get_counts(files, "database")
    labels = [1 for x in pos] + [0 for x in neg]

    sss = SSS(n_splits=1, test_size=0.2, random_state=13)

    scoreA_total = 0.0
    scoreB_total = 0.0
    scoreC_total = 0.0
    for indices in sss.split(arrays, labels):
        train_data = [arrays[x] for x in indices[0]]
        train_labels = [labels[x] for x in indices[0]]
        test_data = [arrays[x] for x in indices[1]]
        test_labels = [labels[x] for x in indices[1]]

    train_data = np.asarray(train_data, dtype='float64')
    train_labels = np.asarray(train_labels)
    train_labels = train_labels.reshape(train_labels.shape[0], 1)

    test_data = np.asarray(test_data, dtype='float64')

    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(train_data)
    Z = scaler.transform(test_data)

    return X, train_labels, Z, test_labels


def data3D():
    pos, neg = setup_files()
    files = pos + neg
    arrays = get_counts(files, "database")
    labels = [1 for x in pos] + [0 for x in neg]

    sss = SSS(n_splits=1, test_size=0.2, random_state=13)

    scoreA_total = 0.0
    scoreB_total = 0.0
    scoreC_total = 0.0
    for indices in sss.split(arrays, labels):
        train_data = [arrays[x] for x in indices[0]]
        train_labels = [labels[x] for x in indices[0]]
        test_data = [arrays[x] for x in indices[1]]
        test_labels = [labels[x] for x in indices[1]]

    train_data = np.asarray(train_data, dtype='float64')
    train_labels = np.asarray(train_labels)
    train_labels = train_labels.reshape(train_labels.shape[0], 1)

    test_data = np.asarray(test_data, dtype='float64')

    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(train_data)
    Z = scaler.transform(test_data)

    X = X.reshape(X.shape + (1,))
    Z = Z.reshape(Z.shape + (1,))

    return X, train_labels, Z, test_labels


def modelA(X, train_labels, Z, test_labels):
    model = Sequential()
    model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}, input_dim = X.shape[1]))
    model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
    model.add(Dropout({{uniform(0,1)}}))
    a = conditional({{choice(['one', 'two', 'three'])}})
    if a == 'two':
        model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}, input_dim = X.shape[1]))
        model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
        model.add(Dropout({{uniform(0,1)}}))
    if a == 'three':
        model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}, input_dim = X.shape[1]))
        model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
        model.add(Dropout({{uniform(0,1)}}))
        model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}, input_dim = X.shape[1]))
        model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
        model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(X, train_labels,
                  batch_size={{choice([5, 10, 15, 32])}},
                  epochs = 100,
                  verbose=0,
                  validation_data=(Z, test_labels))
    score, acc = model.evaluate(Z, test_labels, verbose=0)
    print "\nParams: " + str(space) + "\n"
    print "Score: " + str(acc)
    with open('paramsA.txt', 'a') as f:
        f.write("Params: "+str(space) + "\n")
        f.write("Score: "+str(acc) + "\n\n")
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def modelB(X, train_labels, Z, test_labels):
    model = Sequential()
    model.add(Conv1D({{choice([5,10,15,20,25,32])}}, {{choice([1,3,5,7,9,11])}},
                    activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}},
                    input_shape=(X.shape[1],1)))
    a = conditional({{choice(['maxPool', 'avgPool' 'noPool'])}})
    if a == 'maxPool':
        model.add(MaxPooling1D(pool_size={{choice([1,2,3,4,5,6,7])}}, strides={{choice([1,2,3,4])}},
                                padding='same'))
    model.add(Dropout({{uniform(0,1)}}))
    b = conditional({{choice([1, 2, 3, 4])}})
    for i in range(b-1):
        model.add(Conv1D({{choice([5,10,15,20,25,32])}}, {{choice([1,3,5,7,9,11])}},
                        activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}},
                        input_shape=(X.shape[1],1)))
        a = conditional({{choice(['maxPool', 'avgPool' 'noPool'])}})
        if a == 'maxPool':
            model.add(MaxPooling1D(pool_size={{choice([1,2,3,4,5,6,7])}},
                                   strides={{choice([1,2,3,4])}}, padding='same'))
        model.add(Dropout({{uniform(0,1)}}))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = {{choice(['adam', 'sgd', 'rmsprop'])}},
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    model.fit(X, train_labels,
                batch_size={{choice([5,10,15,32])}},
                epochs=1,
                verbose=0,
                validation_data=(Z, test_labels))
    score, acc = model.evaluate(Z, test_labels, verbose=0)
    print "\nParams: " + str(space) + "\n"
    print "Score: " + str(acc)
    with open('paramsB.txt', 'a') as f:
        f.write("Params: "+str(space) + "\n")
        f.write("Score: "+str(acc) + "\n\n")
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def get_complete_search_space(model):
    model_string = inspect.getsource(model)
    model_string = remove_imports(model_string)
    parts = hyperparameter_names(model_string)
    hyperopt_params = get_hyperparameters(model_string)
    space = get_hyperopt_space(parts, hyperopt_params, verbose=False)
    return space



def runA():
    trialsA = Trials()
    best_runA, best_modelA, space = optim.minimize(model=modelA,
                                            data = data2D,
                                            algo=tpe.suggest,
                                            max_evals=2000,
                                            trials=trialsA,
                                            eval_space=True,
                                            return_space=True,
                                            verbose=False)
    X, Y, Z, Zprime = data2D()
    score, acc = best_modelA.evaluate(Z, Zprime, verbose=0)
    with open('paramsA.txt', 'a') as f:
        f.write("=========\tBEST\t==========\n")
        f.write("Params: "+str(best_runA) + "\n")
        f.write("Score: "+str(acc)+"\n\n")
        f.write(str(get_complete_search_space(modelA)))

    plot_model(best_modelA, to_file="ModelsA/best.png", show_shapes=True,
                            show_layer_names=True)

    for t, trial in enumerate(trialsA):
        plot_model(trial['result']['model'], to_file="ModelsA/model%d.png"%t,
                    show_shapes=True, show_layer_names=True)

def runB():
    trialsB = Trials()
    best_runB, best_modelB, space = optim.minimize(model=modelB,
                                            data = data3D,
                                            algo=tpe.suggest,
                                            max_evals=2,
                                            trials=trialsB,
                                            eval_space=True,
                                            return_space=True,
                                            verbose=False)

    X, Y, Z, Zprime = data3D()
    score, acc = best_modelB.evaluate(Z, Zprime, verbose=0)
    with open('paramsB.txt', 'a') as f:
        f.write("=========\tBEST\t==========\n")
        f.write("Params: "+str(best_runB) + "\n")
        f.write("Score: "+str(acc)+"\n\n")
        f.write(str(get_complete_search_space(modelB)))

    plot_model(best_modelB, to_file="ModelsB/best.png", show_shapes=True, show_layer_names=True)

    for t, trial in enumerate(trialsB):
        plot_model(trial['result']['model'], to_file="ModelsB/model%d.png"%t,
                    show_shapes=True, show_layer_names=True)
