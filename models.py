"""
Contains defintions for machine learning models.
"""

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier as SGDC
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.utils import to_categorical
from utils import flatten, make3D


def neural_network(input_data, validate=True):
    """
    Constructs, compiles, trains and makes predictions with a neural network.
    Returns the predicted values for "predict". If args[0] is true the
    predictions will be 0's and 1's if not the predictions will be floats
    between 0.0 and 1.0 with values closer to 0.0 and 1.0 indicating a higher
    probability of the prediction being correct.
    """

    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    train_classes = np.unique(np.asarray(y_train))
    num_classes = len(list(train_classes))

    y_train = to_categorical(y_train)

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
    return output


def support_vector_machine(input_data, kernel='linear', C=1, feature_names=None,
                           num_features=10, validate=True):
    """
    Fits, trains, and makes predictions with a support vector machine. Returns
    the predicted values.

    Args:
        input_data (tuple):     x_train, y_train, x_test
        kernel (str):           The kernel to be used by the SVM
        C (int or float):       The regularization parameter for the SVM
        feature_names (list):   The names of every feature in input_data, if
                                give, a sorted [high to low] list of the most
                                important features used to make predictions is
                                also returned.
        num_features (int):     How many of the important features to return, if
                                zero all of the features are returned.

    Returns:
        list: The predicted classes for each sample in x_test
        or
        tuple: prdicted classes, top features
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
        coefs = np.absolute(np.argsort(model.coef_.ravel()))
        top_features = feature_names[coefs[-num_features:]]
        output = (output_data, top_features)
    else:
        output = output_data

    return output


#First Optimization
# def random_forest_validation(input_data,n_estimators=5,criterion='gini',
#                              max_features=None,max_depth=None,
#                              min_samples_split=5,min_samples_leaf=1,
#                              min_weight_fraction_leaf=0.1,
#                              max_leaf_nodes=10,min_impurity_decrease=0,
#                              bootstrap=True, n_jobs=-1):

#Second Optimization
def random_forest(input_data, n_estimators=50, criterion='entropy',
                  max_features='log2', max_depth=100, min_samples_split=2,
                  min_samples_leaf=1, min_weight_fraction_leaf=0.01,
                  max_leaf_nodes=25, min_impurity_decrease=0.001, n_jobs=-1,
                  bootstrap=False, feature_names=None, num_features=10,
                  validate=True):
    """
    Fits, trains, and evaluates a random forest learning, returns an accuracy.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)

    kwargs = {'n_estimators': n_estimators, 'criterion': criterion,
              'max_features': max_features, 'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf':min_samples_leaf,
              'min_weight_fraction_leaf': min_weight_fraction_leaf,
              'max_leaf_nodes':max_leaf_nodes,
              'min_impurity_decrease': min_impurity_decrease,
              'bootstrap': bootstrap, 'n_jobs': n_jobs}

    model = RandomForestClassifier(**kwargs)
    model.fit(x_train, y_train)
    if validate:
        output_data = model.score(x_test, y_test)
    else:
        output_data = model.predict(x_test)

    if feature_names is not None:
        importances = np.argsort(model.feature_importances_.ravel())
        top_features = feature_names[importances[-num_features:]]
        output = (output_data, top_features)
    else:
        output = output_data
    return output


# The parameters in the below functions were found by performing a grid search.
# The results from the grid search are in ~/Data/sgdc_parameters/
def kmer_split_sgd(input_data):
    """
    Stochastic gradient descent model with params optimized for kmer data on the
    lupolova et al split dataset
    """
    model = SGDC(loss='log', n_jobs=-1, eta0=1.0,
                 learning_rate='invscaling', penalty='none', tol=0.001,
                 alpha=100000000.0)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def kmer_mixed_sgd(input_data):
    """
    Stochastic gradient descent model with params optimized for kmer data on the
    lupolova et al mixed dataset
    """
    model = SGDC(loss='squared_hinge', n_jobs=-1, penalty='none',
                 tol=0.001, alpha=10000000.0)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def genome_split_sgd(input_data):
    """
    Stochastic gradient descent model with params optimized for genome region
    data on the lupolova split data set.
    """
    model = SGDC(loss='hinge', n_jobs=-1, eta0=0.1,
                 learning_rate='invscaling', penalty='l1', tol=0.001,
                 alpha=0.01)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def genome_mixed_sgd(input_data):
    """
    Stochastic gradient descent model with params optimized for genome region
    data on the lupolova mixed dataset.
    """
    model = SGDC(loss='log', n_jobs=-1, eta0=0.1,
                 learning_rate='invscaling', penalty='l1', tol=0.001,
                 alpha=0.001)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])
