"""
A collection of methods that perform feature selection on the training and
testing data.

Each method has a positional argument (input_data) and a named argument
(feature_names). input_data should be a tuple containing (x_train, y_train,
x_test, y_test). feature_names should be a list containing the names of all the
features in each sample. If feature_names is given the features that are removed
from input_data by the feature selection will also be removed from feature_names

All of the methods return input_data with some features removed from x_train and
x_test based on the conditions specified by the method and by the parameters
passed to the method. If feature_names is specified an updated version of itself
is also returned.
"""

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.feature_selection import SelectPercentile, f_classif, RFE, RFECV
from sklearn.svm import SVC
from utils import flatten, make3D
import pandas as pd
import numpy as np


def variance_threshold(input_data, feature_names, threshold=0.16):
    """
    Removes all features from x_train and x_test whose variances in x_train is
    less than threshold. Uses scikit-learn's VarianceThreshold If feature_names
    is given it is also returned with any features removed from x_train and
    x_test also removed from feature_names.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        feature_names (list):   The names of all features before selection or
                                None.
        threshold (float):      Lower limit of variance for a feature to be kept

    Returns:
        tuple: (x_train, y_train, x_test, y_test), feature_names
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = VarianceThreshold(threshold=threshold)
    x_train = feature_selector.fit_transform(x_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    output_data = (x_train, y_train, x_test, y_test)

    if feature_names is not None:
        mask = feature_selector.get_support()
        feature_names = feature_names[mask]

    return output_data, feature_names


def remove_constant(input_data, feature_names):
    """
    Removes all features from x_train and x_test that are completely constant in
    x_train. If feature_names is given it is also returned with any
    features removed from x_train and x_test also removed from feature_names.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        feature_names (list):   The names of all features before selection or
                                None

    Returns:
        tuple: (x_train, y_train, x_test, y_test), feature_names
    """
    x_train = pd.DataFrame(input_data[0])
    x_train = x_train.loc[:, x_train.var() != 0.0]
    x_test = pd.DataFrame(input_data[2])
    x_test = x_test[list(x_train)]

    output_data = (np.asarray(x_train), input_data[1], np.asarray(x_test),
                   input_data[3])

    if feature_names is not None:
        feature_names = feature_names[list(x_train)]

    return output_data, feature_names


def select_k_best(input_data, feature_names, score_func=f_classif, k=500):
    """
    Selects the k best features in x_train, removes all others from x_train and
    x_test. Selects the best features by using the score function score_func
    and scikit-learn's SelectKBest. If feature_names is given it is also
    returned with any features removed from x_train and x_test also removed from
    feature_names.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        feature_names (list):   The names of all features before selection or
                                None.
        score_func (function):  The score function to be passed to SelectKBest
        k (int):                How many features to keep.

    Returns:
        tuple: (x_train, y_train, x_test, y_test), feature_names
    """

    if score_func == f_classif:
        input_data, feature_names = remove_constant(input_data, feature_names)

    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = SelectKBest(score_func=score_func, k=k)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    output_data = (x_train, y_train, x_test, y_test)
    if feature_names is not None:
        mask = feature_selector.get_support()
        feature_names = feature_names[mask]

    return output_data, feature_names


def select_percentile(input_data, feature_names, score_func=chi2, percentile=5):
    """
    Selects the percentile best features in x_train, removes the rest of the
    features from x_train and x_test. Selects the best features by using the
    score function score_func and scikit-learn's SelectPercentile. If
    feature_names is given it is also returned with any features removed from
    x_train and x_test also removed from feature_names.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        feature_names (list):   The names of all features before selection or
                                None.
        score_func (function):  The score function to be passed to SelectKBest
        percentile (int):       Percentile of features to keep.

    Returns:
        tuple: (x_train, y_train, x_test, y_test), feature_names
    """
    if score_func == f_classif:
        input_data, feature_names = remove_constant(input_data, feature_names)

    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = SelectPercentile(score_func=score_func,
                                        percentile=percentile)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    output_data = (x_train, y_train, x_test, y_test)

    if feature_names is not None:
        mask = feature_selector.get_support()
        feature_names = feature_names[mask]

    return output_data, feature_names


def recursive_feature_elimination(input_data, feature_names,
                                  estimator=SVC(kernel='linear'),
                                  n_features_to_select=None, step=0.1):
    """
    Recursively eliminates features from x_train and x_test using
    scikit-learn's RFE, see documentation:
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    If feature_names is given it is also returned with any features from
    x_train and x_test also removed from feature_names.

    Args:
        input_data (tuple):                   x_train, y_train, x_test, y_test
        feature_names (list):                 The names of all features before
                                              feature selection or None.
        estimator (object):                   Passed to RFE, see documentation
        n_features_to_select (int or None):   Passed to RFE, see documentation
        step (int or float):                  Passed to RFE, see documentation

    Returns:
        tuple: (x_train, y_train, x_test, y_test), feature_names
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = RFE(estimator, n_features_to_select, step)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    output_data = (x_train, y_train, x_test, y_test)

    if feature_names is not None:
        mask = feature_selector.get_support()
        feature_names = feature_names[mask]

    return output_data, feature_names


def recursive_feature_elimination_cv(input_data, feature_names, step=0.1, cv=3,
                                     estimator=SVC(kernel='linear')):
    """
    Recursively elinates features from x_train and x_test with cross
    validation, uses scikit-learn's RFECV see documentation:
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    If feature_names is given it is also returned with any features from
    x_train and x_test also removed from feature_names.

    Args:
        input_data (tuple):     x_train, y_train, x_test, y_test
        feature_names:          The names of all features before feature
                                selection or None.
        estimator (object):     Passed to RFECV, see documentation
        step (int or float):    Passed to RFECV, see documentation
        cv (int):               Passed to RFECV, see documentation

    Returns:
        tuple: (x_train, y_train, x_test, y_test), feature_names
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = RFECV(estimator, step, cv)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    output_data = (x_train, y_train, x_test, y_test)

    if feature_names is not None:
        mask = feature_selector.get_support()
        feature_names = feature_names[mask]

    return output_data, feature_names
