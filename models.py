"""
A collection of methods containing machine learning models.

Each method has a positional argument (input_data) and a named argument
(validate).

Input data should be a tuple containing (x_train, y_train, x_test, y_test)
where x_train is a 2D array of the shape (number of samples, number of
features) containing the training data, y_train is 1D array of the shape
(number of samples,) containing the classification labels for the training
data, x_test is 2D array of the shape (number of test samples, number of
features) containing the test data, and y_test is either a 1D array of the
shape (number of test samples,) containing the classification labels for the
test data or is an empty array in the case where you are not validating the
model. Validate should be a bool. If validate is True, the method will return
an accuracy score representing the percentage of samples in x_test that were
corretly classified and y_test must be given. If validate is False, the method
will return a list containing the predicted classification for each sample in
x_test and y_test is ignored.
"""

from builtins import zip
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.utils import to_categorical
from utils import flatten, make3D, convert_well_index


def neural_network(input_data, feature_names=None, validate=True):
    """
    Constructs, compiles, trains and tests/makes predictions with a neural
    network.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        feature_names (list):   Ignored, here for compatability.
        validate (bool):        If True, an accuracy is returned, if False a
                                list of predictions for x_test is returned.

    Returns:
        float: model accuracy
        or
        list: predicted classifications for x_test.
    """
    feature_names = None

    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    train_classes = np.unique(np.asarray(y_train))
    num_classes = len(list(train_classes))

    y_train = to_categorical(y_train)
    if validate:
        y_test = to_categorical(y_test)

    if len(x_train.shape) == 2:
        x_train = make3D(x_train)
        x_test = make3D(x_test)
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape=(x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    if validate:
        evaluation = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
        output = evaluation[1]
    else:
        output = model.predict(x_test)
    return (output, feature_names)


def support_vector_machine(input_data, kernel='linear', C=1,
                           feature_names=None, validate=True):
    """
    Fits, trains, and tests/makes predictions with a support vector machine.

    Args:
        input_data (tuple):     x_train, y_train, x_test
        kernel (str):           The kernel to be used by the SVM
        C (int or float):       The regularization parameter for the SVM
        feature_names (list):   The names of every feature in input_data, if
                                given, a sorted [high to low] list of the most
                                important features used to make predictions is
                                also returned.
        validate (bool):        If True a model accuracy is returned, if False
                                a list of predicted classifications for x_test
                                is returned.

    Returns:
        list: Predicted classifications for each x_test
        or
        float: Model accuracy
        or
        tuple: accuracy/predictions, features ranked by importance
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(x_train, y_train)
    if validate:
        output_data = model.score(x_test, y_test)
    else:
        output_data = model.predict(x_test)

    if feature_names is not None:
        coefs = model.coef_.ravel()
        absolute_coefs = np.absolute(coefs)
        absolute_coefs = [float(x) for x in absolute_coefs]
        feature_names = [convert_well_index(x) for x in feature_names]
        features_coefs = dict(list(zip(feature_names, absolute_coefs)))
        output = (output_data, features_coefs)
    else:
        output = (output_data, None)

    return output


def random_forest(input_data, n_estimators=50, feature_names=None,
                  validate=True):
    """
    Fits, trains, and tests/makes predictions with s a random forest
    classifier.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        n_estimators (int):     How many trees to use in the forest.
        feature_names (list):   The names of every feature in input_data, if
                                given, a sorted [high to low] list of the most
                                important features used to make predictions is
                                also returned.
        validate (bool):        If True a model accuracy is returned, if False
                                a list of predicted classifications for x_test
                                is returned.

    Returns:
        list: Predicted classifications for each x_test
        or
        float: Model accuracy
        or
        tuple: accuracy/predictions, features ranked by importance
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)

    kwargs = {'n_estimators': n_estimators, 'criterion': 'entropy',
              'max_features': 'log2', 'max_depth': 100, 'min_samples_split': 2,
              'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.01,
              'max_leaf_nodes': 25, 'min_impurity_decrease': 0.001,
              'bootstrap': False, 'n_jobs': -1}

    model = RandomForestClassifier(**kwargs)
    model.fit(x_train, y_train)
    if validate:
        output_data = model.score(x_test, y_test)
    else:
        output_data = model.predict(x_test)

    if feature_names is not None:
        importances = model.feature_importances_.ravel()
        importances = [float(x) for x in importances]
        feature_names = [convert_well_index(x) for x in feature_names]
        features_importances = dict(list(zip(feature_names, importances)))
        output = (output_data, features_importances)
    else:
        output = (output_data, None)
    return output
