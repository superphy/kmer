import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.pooling import AveragePooling1D
from keras.layers.convolutional import Conv1D
from sklearn import svm
from utils import flatten, make3D


def get_methods():
    output = {'nn_validation': neural_network_validation,
              'svm_validation': support_vector_machine_validation,
              'nn': neural_network,
              'svm': support_vector_machine}
    return output


def neural_network_validation(x_train, y_train, x_test, y_test, *args):
    """
    Constructs, compiles, trains, and tests a neural network.
    Returns the accuracy of the model.
    """
    if len(x_train.shape) == 2:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape = (x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    evaluation = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
    return evaluation[1]


def neural_network(x_train, y_train, predict, *args):
    """
    Constructs, compiles, trains and makes predictions with a neural network.
    Returns the predicted values for "predict". If args[0] is true the
    predictions will be 0's and 1's if not the predictions will be floats
    between 0.0 and 1.0 with values closer to 0.0 and 1.0 indicating a higher
    probability of the prediction being correct.
    If args is not provided, default values will be used.
    """
    binarize = True
    if args:
        binarize = args[0]
    if len(x_train.shape) == 2:
        x_train = make3D(x_train)
        x_test = make3D(x_test)
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape = (x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    prediction = model.predict(predict)
    if binarize:
        prediction = np.where(prediction > 0.5, 1, 0)
    return prediction


def support_vector_machine_validation(x_train, y_train, x_test, y_test, *args):
    """
    Fits, trains, and test a support vector machine.
    Returns the accuracy of the model.
    """
    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)

    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def support_vector_machine(x_train, y_train, predict, *args):
    """
    Fits, trains, and makes predictions with a support vector machine.
    Returns the predicted values for "predict"
    """
    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model.predict(predict)
