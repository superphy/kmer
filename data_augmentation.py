from imblearn.over_sampling import SMOTE, ADASYN
from feature_selection import flatten, make3D
import numpy as np
import random
from utils import shuffle
import inspect


def __augment_data_naive_helper(data, desired_samples):
    """
    Helper method for augment_data_naive
    """
    indices = np.random.randint(data.shape[0], size=2*desired_samples)
    temp = np.split(data[indices,:], desired_samples)
    new_data = []
    for elem in temp: new_data.append(elem.mean(axis=0))
    new_data = np.asarray(new_data)
    data = np.vstack((data, new_data))
    return data


def augment_data_naive(input_data, desired_samples=50):
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

    classes = np.unique(y_train)
    output = []
    labels = []
    for c in classes:
        indices = np.where(y_train==c, True, False)
        samples = x_train[indices]
        new_samples = __augment_data_naive_helper(samples, desired_samples)
        output.append(new_samples)
        labels.append(c)
    x_train, y_train = shuffle(output, labels)
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
    ratio = {}
    for c in range(len(classes)):
        ratio[classes[c]] = counts[c]+desired_samples
    smote = SMOTE(ratio=ratio)
    x_train, y_train = smote.fit_sample(x_train, y_train)
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
    classes = np.unique(y_train)
    ratio = {}
    for c in range(len(classes)):
        ratio[classes[c]] = counts[c]+desired_samples
    adasyn = ADASYN(ratio=ratio)
    x_train, y_train = adasyn.fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)


def augment_data_noise(input_data, desired_samples=50):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    classes = np.unique(y_train)
    output = []
    labels = []
    for c in classes:
        indices = np.where(y_train==c, True, False)
        samples = x_train[indices]
        new_samples = samples[np.random.choice(samples.shape[0], desired_samples)]
        new_samples = np.random.randn(new_samples.shape[1]) + new_samples
        samples = np.vstack((samples, new_samples))
        output.append(samples)
        labels.append(c)
    output = np.asarray(output)
    labels = np.asarray(labels)
    x_train, y_train = shuffle(np.asarray(output), labels)
    return (x_train, y_train, x_test, y_test)


def balance_data_smote(input_data):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    samples = max(np.bincount(y_train))
    classes = np.unique(y_train)
    ratio = {}
    for c in classes:
        ratioi[c] = samples
    x_train, y_train = SMOTE(ratio=ratio).fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)

def balance_data_adasyn(input_data):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    samples = max(np.bincount(y_train))
    classes = np.unique(y_train)
    ratio = {}
    for c in classes:
        ratio[c] = samples
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
    output = []
    labels = []
    for c in classes:
        indices = np.where(y_train==c, True, False)
        samples = x_train[indices]
        desired_samples = largest_count - samples.shape[0]
        if desired_samples > 0:
            samples = __augment_data_naive_helper(samples, desired_samples, choice)
        output.append(samples)
        labels.append(c)

    x_train, y_train = shuffle(output, labels)
    return (x_train, y_train, x_test, y_test)

def balance_data_noise(input_data):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    counts = np.bincount(y_train)
    largest_count = max(counts)
    classes = np.unique(y_train)
    output = []
    labels = []
    for c in classes:
        indices = np.where(y_train==c, True, False)
        samples = x_train[indices]
        desired_samples = largest_count - samples.shape[0]
        if desired_samples > 0:
            new_samples = samples[np.random.choice(samples.shape[0], desired_samples)]
            new_samples = np.random.randn(samples.shape[1]) + new_samples
            samples = np.vstack((samples, new_samples))
        output.append(samples)
        labels.appends(c)

    x_train, y_train = shuffle(samples, labels)
    return (x_train, y_train, x_test, y_test)
