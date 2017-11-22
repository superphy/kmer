from imblearn.over_sampling import SMOTE, ADASYN
from feature_selection import flatten, make3D
import numpy as np
import random
from utils import shuffle


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


def augment_data_naive(input_data, desired_samples=50, choice=2):
    """
    Augments data by grabbing x random samples from a class and
    averaging their values to create another sample of the same class.

    args:   A tuple where the first value is the number of new samples to
            add to each class, and the second value is the number of
            original samples to use when creating a new sample.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    temp = np.asarray(y_train, dtype='bool')
    x_pos = __augment_data_naive_helper(x_train[temp], desired_samples, choice)
    temp = np.invert(temp)
    x_neg = __augment_data_naive_helper(x_train[temp], desired_samples, choice)
    x_train, y_train = shuffle(x_pos, x_neg, 1, 0)
    return (x_train, y_train, x_test, y_test)


def augment_data_smote(input_data, desired_samples=50):
    """
    Augments data using the SMOTE algorithm. For more information see the
    documentation:  http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html

    Will probably give a user warning stating: "The number of smaples in
    class x will be larger than the number of samples in the majority class"
    but we can ignore this since we are using SMOTE to augment data, not to
    correct for imbalanced data.

    args:   A tuple where the first value is the number of smaples to be
            added to each class, the second value is the label for the
            lexicographically smaller of the classes, and the third value
            is the label for the remaining class.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    classes = np.unique(y_train)
    a = counts[0]+desired_samples
    b = counts[1]+desired_samples
    ratio = {classes[0]:a, classes[1]:b}
    x_train, y_train = SMOTE(ratio=ratio).fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)


def augment_data_adasyn(input_data, desired_samples=50):
    """
    Augments data using the ADASYN algorithm. For more information see the
    documentation:  http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.ADASYN.html

    Will probably give a user warning stating: "The number of smaples in
    class x will be larger than the number of samples in the majority class"
    but we can ignore this since we are using ADASYN to augment data, not to
    correct for imbalanced data.

    args:   A tuple where the first value is the number of smaples to be
            added to each class, the second value is the label for the
            lexicographically smaller of the classes, and the third value
            is the label for the remaining class.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    vals = np.unique(y_train)
    ratio = {vals[0]:counts[0]+desired_samples, vals[1]:counts[1]+desired_samples}
    x_train, y_train = ADASYN(ratio=ratio).fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)


def augment_data_noise(input_data, desired_samples=50):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    clasess = np.unique(y_train)
    class1 = np.where(y_train==classes[0], True, False)
    class2 = np.invert(class1)
    x_pos = x_train[class1]
    x_neg = x_train[class2]
    new_x_pos = x_pos[np.random.choice(x_pos.shape[0], desired_samples)]
    new_x_neg = x_neg[np.random.choice(x_neg.shape[0], desired_samples)]
    new_x_pos = np.random.randn(x_pos.shape[1]) + new_x_pos
    new_x_neg = np.random.randn(x_neg.shape[1]) + new_x_neg
    x_pos = np.vstack((x_pos, new_x_pos))
    x_neg = np.vstack((x_neg, new_x_neg))
    x_train, y_train = shuffle(x_pos, x_neg, 1, 0)
    return (x_train, y_train, x_test, y_test)


def balance_data_smote(input_data):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    classes = np.unique(y_train)
    ratio = {classes[0]:max(counts), classes[1]:max(counts)}
    x_train, y_train = SMOTE(ratio=ratio).fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)

def balance_data_adasyn(input_data):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    classes = np.unique(y_train)
    ratio = {classes[0]:max(counts), classes[1]:max(counts)}
    x_train, y_train = ADASYN(ratio=ratio).fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)

def balance_data_naive(input_data, choice=2):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    largest_count = max(counts)
    classes = np.unique(y_train)

    class1 = np.where(y_train==classes[0], True, False)
    x_pos = x_train[class1]
    desired_samples = largest_count - x_pos.shape[0]
    if desired_samples > 0:
        x_pos = __augment_data_naive_helper(x_pos, desired_samples, choice)

    class2 = np.invert(class1)
    x_neg = x_train[class2]
    desired_samples = largest_count - x_neg.shape[0]
    if desired_samples > 0:
        x_neg = __augment_data_naive_helper(x_neg, desired_samples, choice)

    x_train, y_train = shuffle(x_pos, x_neg, 1, 0)
    return (x_train, y_train, x_test, y_test)

def balance_data_noise(input_data):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    largest_count = max(counts)
    largest_count = max(counts)
    classes = np.unique(y_train)
    class1 = np.where(y_train==classes[0], True, False)
    x_pos = x_train[class1]
    desired_samples = largest_count - x_pos.shape[0]
    if desired_samples > 0:
        new_x_pos = x_pos[np.random.choice(x_pos.shape[0], desired_samples)]
        x_pos = np.vstack((x_pos, new_x_pos))

    class2 = np.invert(class1)
    x_neg = x_train[class2]
    desired_samples = largest_count - x_neg.shape[0]
    if desired_samples > 0:
        new_x_neg = x_neg[np.random.choice(x_pos.shape[0], desired_samples)]
        x_neg = np.vstack((x_neg, new_x_neg))

    x_train, y_train = shuffle(x_pos, x_neg, 1, 0)

    return (x_train, y_train, x_test, y_test)
