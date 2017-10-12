import subprocess
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, RepeatVector
from keras.layers import Flatten, Reshape, Input, concatenate, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.utils import plot_model
from sklearn.model_selection import StratifiedShuffleSplit as ssSplit
import numpy as np
import os
import lmdb
import sys



def start(filename, k, limit, env, txn, data):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Performs a kmer count on filename, counting kmers with a length of k and
    removing any kmer that has a count less than limit. Resets the master
    database data and then writes each kmer as a key with value -1 to data.
    Creates a new database called filename, writes each kmer/count pair to the
    new databse as a key value pair.
    """
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30', '-C',
            '%s' % filename, '-o', 'test.jf', '-L', '%d' % limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    txn.drop(data, delete=False)
    string = '%s'%filename
    current = env.open_db(string, txn=txn)

    for line in arr:
        txn.put(line[0], line[1])
        txn.put(line[0], line[1], db=current)



def firstpass(filename, k, limit, env, txn):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Performs a kmer count on filename, counting kmers with a length of k and
    removing any kmer that has a count less than limit. Creates a new database
    called filename and writes each kmer/count pair from the kmer count to the
    new database. Only writes kmers that are already present in the master
    database that txn points to. Removes any kmer from the master database that
    is not present in filename.
    """
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30', '-C',
            '%s' % filename, '-o', 'test.jf', '-L', '%d' % limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    txn.drop(current, delete= False)

    for line in arr:
        if txn.get(line[0], default=False):
            txn.put(line[0], line[1], overwrite=True, dupdata=False)

    cursor = txn.cursor()
    for key, value in cursor:
        if value == '-1':
            txn.delete(key)
        else:
            txn.put(key, value, db=current)
            txn.put(key, '-1')



def second_start(filename, k, env, txn, data):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Resets the master database so that it matches the database named filename.
    """
    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    txn.drop(data, delete=False)
    cursor = txn.cursor(db=current)
    for key, val in cursor:
        txn.put(key, val)



def secondpass(filename, k, env, txn):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Removes every kmer from the database named filename that is not present in
    the master database.
    """
    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    cursor = txn.cursor(db = current)

    for key, val in cursor:
        if not txn.get(key, default=False):
            txn.delete(key, val, db = current)



def print_status(counter, total):
    """
    Outputs a progress bar.
    """
    percent = (counter*100)/total
    sys.stdout.write('\r')
    sys.stdout.write("[%-44s] %d%%" % ('='*((percent*44)/100), percent))
    sys.stdout.flush()



def set_up_files(filepath):
    """
    Helper method for running kmer_prediction.py from the command line.

    Takes a path to a directory, returns a list of the complete paths to each
    file in the directory
    """
    if not filepath[-1] == '/':
        filepath += '/'
    return [filepath + x for x in os.listdir(filepath)]



def setup_data(files, k, limit, env, txn, data):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Takes a list of paths to fasta files, a kmer length, a lower limit on how
    many times a kmer needs to occur in order for it to be output, and an lmdb
    environment, transaction and database.
    """
    counter = 0
    total = len(files)
    start(files[0], k, limit, env, txn, data)
    temp = files.pop(0)
    counter += 1
    for filename in files:
        print_status(counter, total)
        firstpass(filename, k, limit, env, txn)
        counter += 1

    files.insert(0, temp)

    print_status(counter, total)
    print ""
    counter = 0
    second_start(files[-1], k, env, txn, data)
    counter += 1
    i = len(files)-2
    while i >= 0:
        print_status(counter, total)
        secondpass(files[i], k, env, txn)
        i-=1
        counter += 1

    print_status(counter, total)
    print "\n"



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



def inception(input_stream):

    parrallel1 = Conv1D(32, 3, activation='relu', padding='same')(input_stream)
    parrallel1 = Dropout(0.5)(parrallel1)

    parrallel2 = Conv1D(15, 3, activation='relu', padding='same')(input_stream)
    parrallel2 = Conv1D(32, 3, activation='relu', padding='same')(parrallel2)
    parrallel2 = Dropout(0.5)(parrallel2)

    parrallel3 = Conv1D(15, 3, activation='relu', padding='same')(input_stream)
    parrallel3 = Conv1D(32, 3, activation='relu', padding='same')(parrallel3)
    parrallel3 = Dropout(0.5)(parrallel3)

    parrallel4 = AveragePooling1D(pool_size=3, strides=1, padding='same')(input_stream)
    parrallel4 = Conv1D(32, 3, activation='relu', padding='same')(parrallel4)
    parrallel4 = Dropout(0.5)(parrallel4)

    output_stream = concatenate([parrallel1,parrallel2,parrallel3,parrallel4],axis=-1)

    return output_stream


def bioinformatics(length):
        model = Sequential()
        model.add(Conv1D(32, 3, input_shape=(length, 1), activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        return model


def make_predictions(train_data, train_labels, test_data, test_labels):
    """
    Helper method for kmer_prediction.run(), should not be used on its own.

    Takes training data and labels, and test data for an svm. If test_lables is
    None returns predictions on test_data, other wise returns a percentage of
    predictions that the svm got correct.
    """
    train_data = np.asarray(train_data, dtype='float64')
    train_labels = np.asarray(train_labels)
    train_labels = train_labels.reshape(train_labels.shape[0], 1)

    test_data = np.asarray(test_data, dtype='float64')

    # modelA = Sequential([
    #         Dense(32, input_dim = train_data.shape[1]),
    #         Activation('sigmoid'),
    #         Dense(1),
    #         Activation('sigmoid')
    #         ])
    modelB = Sequential([
            Conv1D(15, 3, activation='relu', input_shape=(train_data.shape[1],1)),
            Conv1D(32, 3, activation='relu'),
            Dropout(0.25),
            Flatten(),
            Dense(1, activation='sigmoid')
            ])

    # inputA = Input(shape=(train_data.shape[1],1))
    #
    # input_streamA = Conv1D(5, 3, activation='relu')(inputA)
    # input_streamA = MaxPooling1D(pool_size=1, strides=1)(input_streamA)
    # input_streamA = Conv1D(10, 3, activation='relu')(input_streamA)
    # input_streamA = Conv1D(15, 3, activation='relu')(input_streamA)
    # input_streamA = MaxPooling1D(pool_size=1, strides=1)(input_streamA)
    # input_streamA = Dropout(0.25)(input_streamA)
    #
    # parrallelA1 = Conv1D(32, 3, activation='relu', padding='same')(input_streamA)
    # parrallelA1 = Dropout(0.5)(parrallelA1)
    #
    # parrallelA2 = Conv1D(15, 3, activation='relu', padding='same')(input_streamA)
    # parrallelA2 = Conv1D(32, 3, activation='relu', padding='same')(parrallelA2)
    # parrallelA2 = Dropout(0.5)(parrallelA2)
    #
    # parrallelA3 = Conv1D(15, 3, activation='relu', padding='same')(input_streamA)
    # parrallelA3 = Conv1D(32, 3, activation='relu', padding='same')(parrallelA3)
    # parrallelA3 = Dropout(0.5)(parrallelA3)
    #
    # parrallelA4 = MaxPooling1D(pool_size=1, strides=1, padding='same')(input_streamA)
    # parrallelA4 = Dropout(0.5)(parrallelA4)
    # parrallelA4 = Conv1D(32, 3, activation='relu', padding='same')(parrallelA4)
    #
    #output_streamA=concatenate([parrallelA1,parrallelA2,parrallelA3,parrallelA4],axis=-1)
    # output_streamA = MaxPooling1D(pool_size=1, strides=1)(output_streamA)
    # output_streamA = Dropout(0.25)(output_streamA)
    # output_streamA = Flatten()(output_streamA)
    # output_streamA = Dense(1, activation='sigmoid')(output_streamA)
    #
    # modelB = Model(inputA, output_streamA)

    # input = Input(shape=(train_data.shape[1],1))
    #
    # input_stream = Conv1D(5, 3, activation='relu')(input)
    # input_stream = BatchNormalization()(input_stream)
    # input_stream = MaxPooling1D(pool_size=3, strides=1)(input_stream)
    # input_stream = Conv1D(10, 3, activation='relu',)(input_stream)
    # input_stream = Conv1D(15, 3, activation='relu')(input_stream)
    # input_stream = BatchNormalization()(input_stream)
    # input_stream = MaxPooling1D(pool_size=3, strides=1)(input_stream)
    # input_stream = Dropout(0.5)(input_stream)
    #
    # output_stream = AveragePooling1D(pool_size=7,strides=1,padding='same')(input_stream)
    # output_stream = Dropout(0.5)(output_stream)
    # output_stream = Flatten()(output_stream)
    # output_stream = Dense(1, activation='sigmoid')(output_stream)
    # modelC = Model(input, output_stream)

    modelC = bioinformatics(train_data.shape[1])



    # modelC = Sequential([
    #         Conv1D(15, 3, activation='relu', input_shape=(train_data.shape[1],1)),
    #         Conv1D(32, 3, activation='relu'),
    #         MaxPooling1D(pool_size=1),
    #         Dropout(0.25),
    #         Flatten(),
    #         Dense(1, activation='sigmoid')
    #         ])
    # modelD = Sequential([
    #         Conv1D(15, 3, activation='relu', input_shape=(train_data.shape[1],1)),
    #         Conv1D(32, 3, activation='relu'),
    #         Dropout(0.25),
    #         Flatten(),
    #         Dense(1, activation='sigmoid')
    #         ])
    # modelE = Sequential([
    #         Dense(64, input_dim = train_data.shape[1]),
    #         Activation('relu'),
    #         Dense(1),
    #         Activation('sigmoid')
    #         ])
    # modelA.compile(optimizer = 'sgd',
    #             loss = 'binary_crossentropy',
    #             metrics = ['accuracy'])
    modelB.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    modelC.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

    # modelD.compile(optimizer = 'adam',
    #             loss = 'binary_crossentropy',
    #             metrics = ['accuracy'])
    # modelE.compile(optimizer = 'sgd',
    #             loss = 'binary_crossentropy',
    #             metrics = ['accuracy'])

    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(train_data)
    Xprime = X.reshape(X.shape + (1,))
    Z = scaler.transform(test_data)
    Zprime = Z.reshape(Z.shape + (1,))

    try:
        # modelA.fit(X, train_labels, epochs=100, batch_size=10, verbose=1)
        modelB.fit(Xprime, train_labels, epochs=10, batch_size=10, verbose=1)
        modelC.fit(Xprime, train_labels, epochs=10, batch_size=10, verbose=1)
        # modelD.fit(Xprime, train_labels, epochs=10, batch_size=10, verbose=1)
        # modelE.fit(X, train_labels, epochs=10, batch_size=10, verbose=0)
        if test_labels:
            test_labels = np.asarray(test_labels)
            test_labels = test_labels.reshape(test_labels.shape[0], 1)
            # scoreA = modelA.evaluate(Z, test_labels, batch_size=32, verbose=1)
            scoreB = modelB.evaluate(Zprime, test_labels, batch_size=10, verbose=1)
            scoreC = modelC.evaluate(Zprime, test_labels, batch_size=10, verbose=1)
            # scoreD = modelD.evaluate(Zprime, test_labels, batch_size=10, verbose=1)
            # scoreE = modelD.evaluate(Z, test_labels, batch_size=32, verbose=0)
            # score,sensitivity,specificity=sensitivity_specificity(results,test_labels)
            #return (scoreA[1], scoreB[1], scoreC[1])#, scoreD[1], scoreE[1])
            return (scoreB[1], scoreC[1])#, scoreD[1])
        else:
            a = modelA.predict(Z)
            # b = modelB.predict(Zprime)
            # c = modelC.predict(Zprime)
            return a
    except (ValueError, TypeError) as E:
        print E
        return -1



def run(k, limit, num_splits, pos, neg, predict):
    env = lmdb.open('database', map_size=int(160e9), max_dbs=400)
    data = env.open_db('master', dupsort=False)

    with env.begin(write=True, db=data) as txn:

        if not predict:
            files = pos + neg
        else:
            files = pos + neg + predict

        #setup_data(files, k, limit, env, txn, data)

        arrays = []
        for f in files:
            array = []
            current = env.open_db('%s'%f, txn = txn)
            cursor = txn.cursor(db=current)
            for key, val in cursor:
                array.append(int(val))
            arrays.append(array)

    env.close()
    print "Done with DB"

    labels=[1 for x in range(len(pos))]+[0 for x in range(len(neg))]

    if not predict:
        sss = ssSplit(n_splits=num_splits, test_size=0.2, random_state=42)

        scoreA_total = 0.0
        scoreB_total = 0.0
        scoreC_total = 0.0
        scoreD_total = 0.0
        scoreE_total = 0.0
        # sensitivity_total = 0.0
        # specificity_total = 0.0
        for indices in sss.split(arrays, labels):
            X = [arrays[x] for x in indices[0]]
            Y = [labels[x] for x in indices[0]]
            Z = [arrays[x] for x in indices[1]]
            ZPrime = [labels[x] for x in indices[1]]

            scores = make_predictions(X,Y,Z,ZPrime)
            print scores
            scoreA_total += scores[0]
            scoreB_total += scores[1]
            # scoreC_total += scores[2]
            # scoreD_total += scores[3]
            # scoreE_total += scores[4]
            # sensitivity_total += sensitivity
            # specificity_total += specificity

        output = (scoreA_total/num_splits, scoreB_total/num_splits)
                    # scoreC_total/num_splits) #, scoreD_total/num_splits,
                    # scoreE_total/num_splits)

    else:
        sss = ssSplit(n_splits=1, test_size = 0.5, random_state=13)

        for indices in sss.split(arrays[:(len(pos)+len(neg))], labels):
            X = [arrays[x] for x in indices[0]]
            X.extend([arrays[x] for x in indices[1]])
            Y = [labels[x] for x in indices[0]]
            Y.extend([labels[x] for x in indices[1]])

        Z = arrays[len(pos) + len(neg):]
        output = make_predictions(X, Y, Z, None)

    return output



def main():
    """
    Handles kmer_prediction.py being run from the command line.
    """
    if len(sys.argv) == 6 or len(sys.argv) == 7:
        k = int(sys.argv[1])
        l = int(sys.argv[2])
        reps = int(sys.argv[3])
        pos = set_up_files(sys.argv[4])
        neg = set_up_files(sys.argv[5])
        if len(sys.argv) == 6:
            output = run(k, l, reps, pos, neg, None)
        else:
            predict = set_up_files(sys.argv[6])
            output = run(k, l, reps, pos, neg, predict)
        print output
    else:
        output = """
        Error: Wrong number of arguments, requires at least 5, at most 6
        First Argument: kmer length
        Second Argument: Minimum kmer count required for a kmer to be output
        Third Argument: Number of times to repeat the training and testing of
                        the model, if greater than 1 returns the average of all
                        runs.
        Fourth Argument: Path to directory containing fasta files for genomes
                         positive for a phenotype.
        Fifth Argument: Path to a directory containing fasta files negative for
                        a phenotype.
        Sixth Argument: (Optional) Path to a directory containing fasta files
                        that you want to predict whether or not they are
                        positive for the phenotype. If supplied, all the files
                        contianed in the fourth and fifth parameter will be used
                        to train the model. If not supplied the files contained
                        in the fourth and fifth argument will be shuffled and
                        80%% will be used to train the model and the rest will
                        be used to test the model.

        Example: python %s 10 5 5 /home/data/pos/ /home/data/neg/
        Counts kmers with length 10 removing any that appear fewer than 5 times
        then performs the training and testing of the model 5 times using the
        fasta files contained in the positive and negative directories.
        """ % sys.argv[0]
        print output



if __name__ == "__main__":
    main()
