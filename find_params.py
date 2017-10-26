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
import sys
import time
import random

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.optim import get_hyperopt_space, hyperparameter_names, get_hyperparameters
from hyperas.utils import remove_imports

from kmer_counter import count_kmers, get_counts

from ecoli_human_bovine import setup_files

from sklearn.model_selection import StratifiedShuffleSplit as SSS

from sklearn.preprocessing import MinMaxScaler

from verify_results import data_us_uk
from get_fasta_from_json import get_fasta_from_json

from presence_absence_data import get_data_and_labels

def data():
    pos, neg = setup_files()
    files = pos + neg
    arrays = get_counts(files, "database2")
    labels = [1 for x in pos] + [0 for x in neg]

    # sss = SSS(n_splits=1, test_size=0.2, random_state=13)
    #
    # scoreA_total = 0.0
    # scoreB_total = 0.0
    # scoreC_total = 0.0
    # for indices in sss.split(arrays, labels):
    #     train_data = [arrays[x] for x in indices[0]]
    #     train_labels = [labels[x] for x in indices[0]]
    #     test_data = [arrays[x] for x in indices[1]]
    #     test_labels = [labels[x] for x in indices[1]]
    #
    # train_data = np.asarray(train_data, dtype='float64')
    # train_labels = np.asarray(train_labels)
    # train_labels = train_labels.reshape(train_labels.shape[0], 1)
    #
    # test_data = np.asarray(test_data, dtype='float64')
    #
    # scaler = MinMaxScaler(feature_range=(-1,1))
    # X = scaler.fit_transform(train_data)
    # Z = scaler.transform(test_data)

    return arrays, labels, None, None

def DATA():
    X_train, Y_train, X_test, Y_test =  get_data_and_labels('binary_table.txt', '', True)
    return X_train, Y_train, X_test, Y_test


def entero_data():
    human, bovine = get_fasta_from_json()

    arrays = human + bovine
    labels = [1 for x in human] + [0 for x in bovine]

    arrays = get_counts(arrays, "kmer_counts")
    return arrays, labels


def modelA(arrays, labels, garbage, garbage1):

    model = Sequential()
    model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}, input_dim = len(arrays[0])))
    model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
    model.add(Dropout({{uniform(0,1)}}))
    a = conditional({{choice(['one', 'two', 'three'])}})
    if a == 'two':
        model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}))
        model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
        model.add(Dropout({{uniform(0,1)}}))
    if a == 'three':
        model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}))
        model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
        model.add(Dropout({{uniform(0,1)}}))
        model.add(Dense({{choice([5,10,15,20,25,30,32,64,128])}}))
        model.add(Activation({{choice(['sigmoid','softmax','relu','tanh', None])}}))
        model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    batch1 = {{choice([5,10,15,32])}}
    batch2 = {{choice([5,10,15,32])}}
    sss = SSS(n_splits=10, test_size=0.2, random_state=13)

    score_total = 0.0
    count = 0.0
    maximum = 0.0

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

        model.fit(X, train_labels, batch_size=batch1,
                  epochs=1,
                  verbose=0,
                  validation_data=(Z, test_labels))
        loss, acc = model.evaluate(Z, test_labels,
                                   batch_size = batch2, verbose=0)
        score_total += acc
        count += 1.0
        if acc > maximum:
            maximum = acc
        if acc <= 0.70:
            break
    score = score_total/count
    print "\nParams: " + str(space) + "\n"
    print "Score: " + str(score)
    with open('paramsA.txt', 'a') as f:
        f.write("Params: "+str(space) + "\n")
        f.write("Score: "+str(score) + "\n")
        f.write("Maximum: "+str(maximum)+"\n\n")
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

# model.add(Dropout({{uniform(0,1)}}))
# b = conditional({{choice([1, 2, 3, 4])}})
# for i in range(b-1):
#     model.add(Conv1D({{choice([5,10,15,20,25,32])}}, {{choice([1,3,5,7,9,11])}},
#                     activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}},
#                     input_shape=(X.shape[1],1)))
#     a = conditional({{choice(['maxPool', 'avgPool' 'noPool'])}})
#     if a == 'maxPool':
#         model.add(MaxPooling1D(pool_size={{choice([1,2,3,4,5,6,7])}},
#                                strides={{choice([1,2,3,4])}}, padding='same'))
#     model.add(Dropout({{uniform(0,1)}}))
# model.add(Dropout({{uniform(0,1)}}))

def modelB(arrays, labels, garbage, garbage1):
    model = Sequential()

    model.add(Conv1D(filters={{choice([5,10,15,20,25,32])}},
                    kernel_size={{choice([1,3,5,7,9,11])}},
                    activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}},
                    input_shape=(len(arrays[0]),1)))

    a = conditional({{choice(['maxPool', 'avgPool', 'noPool'])}})

    pool, stride={{choice([((x/7)+1,(x%7)+1) for x in range(0,49) if (x%7)+1 <= (x/7)+1])}}
    if a == 'maxPool':
        model.add(MaxPooling1D(pool_size=pool,strides=stride,padding='same'))
    elif a == "avgPool":
        model.add(AveragePooling1D(pool_size=pool,strides=stride,padding='same'))

    model.add(Dropout({{uniform(0,1)}}))

    if conditional({{choice(['one', 'two'])}}) == "two":
        model.add(Conv1D(filters={{choice([5,10,15,20,25,32])}},
                        kernel_size={{choice([1,3,5,7,9,11])}},
                        activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}}))

        a = conditional({{choice(['maxPool', 'avgPool', 'noPool'])}})

        pool, stride={{choice([((x/7)+1,(x%7)+1) for x in range(0,49) if (x%7)+1 <= (x/7)+1])}}
        if a == 'maxPool':
            model.add(MaxPooling1D(pool_size=pool,strides=stride,padding='same'))
        elif a == "avgPool":
            model.add(AveragePooling1D(pool_size=pool,strides=stride,padding='same'))

        model.add(Dropout({{uniform(0,1)}}))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = {{choice(['adam', 'sgd', 'rmsprop'])}},
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    batch1 = {{choice([5,10,15,32])}}
    batch2 = {{choice([5, 10, 15, 32])}}
    sss = SSS(n_splits=10, test_size=0.2, random_state=13)

    score_total = 0.0
    count = 0.0
    maximum = 0.0
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

        model.fit(X, train_labels, batch_size=batch1,
                  epochs=100,
                  verbose=0)
        loss, acc = model.evaluate(Z, test_labels,
                                   batch_size=batch2, verbose=0)
        score_total += acc
        count += 1.0
        if acc > maximum:
            maximum = acc
        if acc <= 0.80:
            break
        if acc >= 0.94:
            model.save('Models/ModelB'+time.strftime("%Y-%m-%d-%H-%M")+'-'+str(acc)+'.h5')
    score = score_total/count
    print "\nParams: " + str(space) + "\n"
    print "Score: " + str(score)
    with open('paramsB.txt', 'a') as f:
        f.write("Params: "+str(space)+"\n")
        f.write("Score: "+str(score)+"\n")
        f.write("Maximum: "+str(maximum)+"\n\n")
    return {'loss': -score, 'status': STATUS_OK, 'model': model}




def modelC(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Conv1D(filters={{choice([5,10,15,20,25,32])}},
                    kernel_size={{choice([1,3,5,7,9,11])}},
                    activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}},
                    input_shape=(X_train.shape[1],1)))

    a = conditional({{choice(['maxPool', 'avgPool', 'noPool'])}})

    pool, stride={{choice([((x/7)+1,(x%7)+1) for x in range(0,49) if (x%7)+1 <= (x/7)+1])}}
    if a == 'maxPool':
        model.add(MaxPooling1D(pool_size=pool,strides=stride,padding='same'))
    elif a == "avgPool":
        model.add(AveragePooling1D(pool_size=pool,strides=stride,padding='same'))

    model.add(Dropout({{uniform(0,1)}}))

    if conditional({{choice(['one', 'two'])}}) == "two":
        model.add(Conv1D(filters={{choice([5,10,15,20,25,32])}},
                        kernel_size={{choice([1,3,5,7,9,11])}},
                        activation={{choice(['relu', 'sigmoid', 'softmax', 'tanh'])}}))

        a = conditional({{choice(['maxPool', 'avgPool', 'noPool'])}})

        pool, stride={{choice([((x/7)+1,(x%7)+1) for x in range(0,49) if (x%7)+1 <= (x/7)+1])}}
        if a == 'maxPool':
            model.add(MaxPooling1D(pool_size=pool,strides=stride,padding='same'))
        elif a == "avgPool":
            model.add(AveragePooling1D(pool_size=pool,strides=stride,padding='same'))

        model.add(Dropout({{uniform(0,1)}}))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = {{choice(['adam', 'sgd', 'rmsprop'])}},
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    batch1 = {{choice([5,10,15,32])}}
    batch2 = {{choice([5, 10, 15, 32])}}
    model.fit(X_train, Y_train, batch_size=batch1, epochs=60, verbose=0)
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch2, verbose=0)
    if acc >= 0.94:
            model.save('Models/ModelC'+time.strftime("%Y-%m-%d-%H-%M")+'-'+str(acc)+'.h5')
    print "\nParams: " + str(space) + "\n"
    print "Score: " + str(acc)
    with open('paramsC.txt', 'a') as f:
        f.write("Params: "+str(space)+"\n")
        f.write("Score: "+str(acc)+"\n")
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
                                            data = data,
                                            algo=tpe.suggest,
                                            max_evals=2,
                                            trials=trialsA,
                                            eval_space=True,
                                            return_space=True,
                                            verbose=False)
    # X, Y, Z, Zprime = data()
    # score, acc = best_modelA.evaluate(Z, Zprime, verbose=0)
    with open('paramsA.txt', 'a') as f:
        f.write("=========\tBEST\t==========\n")
        f.write("Params: "+str(best_runA) + "\n")
        # f.write("Score: "+str(acc)+"\n\n")
        f.write(str(get_complete_search_space(modelA)))

    filename = 'Models/ModelA'+time.strftime("%Y-%m-%d-%H-%M")

    plot_model(best_modelA, to_file=filename+'.png',
               show_shapes=True,
               show_layer_names=True)
    best_modelA.save(filename+'.h5')

    with open(filename+'.json', 'w') as f:
        f.write(best_modelA.to_json())

def runB():
    trialsB = Trials()
    best_runB, best_modelB, space = optim.minimize(model=modelB,
                                            data = entero_data,
                                            algo=tpe.suggest,
                                            max_evals=500,
                                            trials=trialsB,
                                            eval_space=True,
                                            return_space=True,
                                            verbose=False)

    with open('paramsB.txt', 'a') as f:
        f.write("=========\tBEST\t==========\n")
        f.write("Params: "+str(best_runB) + "\n")
        f.write(str(get_complete_search_space(modelB)))

    filename = 'Models/ModelB'+time.strftime("%Y-%m-%d-%H-%M")

    plot_model(best_modelB, to_file=filename+'.png',
               show_shapes=True,
               show_layer_names=True)

    best_modelB.save(filename+'.h5')

    with open(filename+'.json', 'w') as f:
        f.write(best_modelB.to_json())

def runC():
    best_run, best_model, space = optim.minimize(model=modelC,
                                                data=DATA,
                                                algo=tpe.suggest,
                                                max_evals=500,
                                                trials=Trials(),
                                                eval_space=True,
                                                return_space=True,
                                                verbose=False)
    with open('paramsC.txt', 'a') as f:
        f.write("=========\tBEST\t==========\n")
        f.write("Params: "+str(best_run) + "\n")
        f.write(str(get_complete_search_space(modelC)))

    filename = 'Models/ModelC'+time.strftime("Y-%m-%d-%H-%M")

    plot_model(best_model, to_file=filename+'.png', show_shapes=True,
               show_layer_names=True)

    best_model.save(filename+'.h5')
    with open(filename+'.json', 'w') as f:
        f.write(best_model.to_json())


if __name__ == "__main__":
    if sys.argv[1] == 'a':
        runA()
    elif sys.argv[1] == 'b':
        runB()
    elif sys.argv[1] == 'c':
        runC()
