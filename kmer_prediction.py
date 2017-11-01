from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.preprocessing import MinMaxScaler
from kmer_counter import count_kmers, get_counts
import numpy as np



def set_up_files(filepath):
    """
    Takes a path to a directory, returns a list of the complete paths to each
    file in the directory
    """
    if not filepath[-1] == '/':
        filepath += '/'
    return [filepath + x for x in os.listdir(filepath)]



def sensitivity_specificity(predicted_values, true_values):
    """
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



def NeuralNet(length):
    model = Sequential()
    model.add(Conv1D(10, 3, activation='relu', input_shape=(length,1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model



def make_predictions(train_data, train_labels, test_data, test_labels):
    train_data = np.asarray(train_data, dtype='float64')
    train_labels = np.asarray(train_labels)
    train_labels = train_labels.reshape(train_labels.shape[0], 1)

    test_data = np.asarray(test_data, dtype='float64')

    model = NeuralNet(train_data.shape[1])

    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(train_data)
    Z = scaler.transform(test_data)

    Xprime = X.reshape(X.shape + (1,))
    Zprime = Z.reshape(Z.shape + (1,))

    try:
        model.fit(Xprime,
                  train_labels,
                  epochs=50,
                  batch_size=10,
                  verbose=0,)
        if test_labels:
            test_labels = np.asarray(test_labels)
            test_labels = test_labels.reshape(test_labels.shape[0], 1)

            score = model.evaluate(Zprime, test_labels, batch_size=10, verbose=0)
            return score[1]
        else:
            return model.predict(Zprime)
    except (ValueError, TypeError) as E:
        print E
        return [-1]


def run(k, limit, num_splits, pos, neg, predict):
    if not predict:
        files = pos + neg
    else:
        files = pos + neg + predict

    # count_kmers( k, limit, files, "database")

    arrays = get_counts(files, "database")

    labels=[1 for x in pos]+[0 for x in neg]

    if not predict:
        sss = SSS(n_splits=num_splits, test_size=0.2, random_state=42)

        score_total = 0.0
        for indices in sss.split(arrays, labels):
            X = [arrays[x] for x in indices[0]]
            Y = [labels[x] for x in indices[0]]
            Z = [arrays[x] for x in indices[1]]
            ZPrime = [labels[x] for x in indices[1]]

            scores = make_predictions(X,Y,Z,ZPrime)
            print scores
            score_total += scores

        output = score_total/num_splits

    else:
        X = arrays[:len(pos) + len(neg)]
        Z = arrays[len(pos) + len(neg):]
        output = make_predictions(X, labels, Z, None)

    return output
