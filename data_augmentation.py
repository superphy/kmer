from imblearn.over_sampling import SMOTE, ADASYN
from feature_selection import flatten, make3D
import numpy as np
import random
from utils import shuffle


def get_methods():
    output = {'naive': augment_data_naive,
              'smote': augment_data_smote,
              'adasyn': augment_data_adasyn}
    return output


def __augment_data_naive_helper(data, desired_samples, x):
    """
    Helper method for augment_data_naive
    """
    indices = np.random.randint(data.shape[0], size=x*desired_samples)
    temp = np.split(data[indices,:], desired_samples)
    new_data = []
    for elem in temp: new_data.append(elem.mean(axis=0))
    new_data = np.asarray(new_data)
    data = np.vstack((data, new_data))
    return data


def augment_data_naive(x_train, y_train, x_test, y_test, args=[50, 2]):
    """
    Augments data by grabbing args[1] random samples from the same class and
    averaging their values to create another sample of the same class. Adds
    args[0] more samples to each class in the data.
    """
    desired_samples = args[0]
    x = args[1]
    temp = np.asarray(y_train, dtype='bool')
    x_pos = __augment_data_naive_helper(x_train[temp], desired_samples, x)
    temp = np.invert(temp)
    x_neg = __augment_data_naive_helper(x_train[temp], desired_samples, x)
    x_train, y_train = shuffle(x_pos, x_neg, 1, 0)
    return x_train, y_train, x_test, y_test


def augment_data_smote(x_train, y_train, x_test, y_test, args=[100]):
    """
    Augments data using the SMOTE algorithm, adds args[0] more samples
    to each classs in the data. For more information see the documentaion:
    http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html
    Will probably give a user warning stating: "The number of smaple in class x
    will be larger than the number of samples in the majority class", but we can
    ignore this since we are using SMOTE to augment data, not to correct for
    imbalanced data.
    """
    desired_samples = args[0]
    ratio = {1:desired_samples, 0:desired_samples}
    new_x, new_y = SMOTE(ratio=ratio).fit_sample(x_train, y_train)
    x_train = np.vstack((x_train, new_x))
    y_train = np.concatenate((y_train, new_y))
    return x_train, y_train, x_test, y_test


def augment_data_adasyn(x_train, y_train, x_test, y_test, args=[200]):
    """
    Augments data using the ADASYN algorithm, adds args[0] more
    samples to each class in the data. For more info see the documentation:
    http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.ADASYN.html
    Will probably give a user warning stating: "The number of smaple in class x
    will be larger than the number of samples in the majority class", but we can
    ignore this since we are using ADASYN to augment data, not to correct for
    imbalanced data.
    """
    desired_samples = args[0]
    ratio = {1:desired_samples, 0:desired_samples}
    new_x, new_y = ADASYN(ratio=ratio).fit_sample(x_train, y_train)
    x_train = np.vstack((x_train, new_x))
    y_train = np.concatenate((y_train, new_y))
    return x_train, y_train, x_test, y_test
