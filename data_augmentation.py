from imblearn.over_sampling import SMOTE, ADASYN
from feature_selection import flatten, make3D
import numpy as np
import random
from utils import shuffle
import inspect


def __augment_data_naive_helper(data, desired_samples):
    """
    Helper method for augment_data_naive

    Args:
        data (ndarray): Data samples all from one class
        desired_samples (int): The number of new samples to be created for this
                               class.

    Returns:
        data (ndarray): The input data with additional samples of the same class
                        added.
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

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test
        desired_samples (int): The number of samples to be added to each class.

    Returns:
        tuple: x_train, y_train, x_test, y_test with samples added to x_train
               and y_train.
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

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test
        desired_samples (int): The number of samples to be added to each class.

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    classes, counts = np.unique(y_train, return_counts=True)
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

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test
        desired_samples (int): The number of samples to be added to each class.

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    classes, counts = np.unique(y_train, return_counts=True)
    ratio = {}
    for c in range(len(classes)):
        ratio[classes[c]] = counts[c]+desired_samples
    adasyn = ADASYN(ratio=ratio)
    x_train, y_train = adasyn.fit_sample(x_train, y_train)
    return (x_train, y_train, x_test, y_test)


def augment_data_noise(input_data, desired_samples=50):
    """
    Augments data by adding random noise to samples.

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test
        desired_samples (int): The number of samples to be added to each class.

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
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
    """
    Uses the SMOTE algorithm to balance data by adding samples to all minority
    classes until each class has the same number of samples.
    See: http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
    """
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
    """
    Uses the ADASYN algorithm to balance data by adding samples to all minority
    classes until each class has the same number of samples.
    See: http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.ADASYN.html

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
    """
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
    """
    Balances data by adding samples to all minority classes until each class has
    the same number of samples. Creates new samples by grabbing x random samples
    from the same class and averaging their values.

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
    """
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
    """
    Balances data by adding samples to all minority classes until each class has
    the same number of samples. Creates new samples by grabbing adding random
    noise to samples.

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test

    Returns:
        tuple: x_train, y_train, x_test, y_test, with samples added to x_train
               and y_train.
    """
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
