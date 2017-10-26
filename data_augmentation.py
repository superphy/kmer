from imblearn.over_sampling import SMOTE, ADASYN
from feature_selection import flatten, make3D
from US_UK_data import shuffle
import numpy as np
import random

def augment_data_naive_helper(data, factor):
    indices = np.random.choice(data.shape[0], 2*data.shape[0]*(factor-1))
    temp = data[indices,:]
    subarrays = np.split(temp, data.shape[0]*(factor-1), axis=0)
    for elem in subarrays:
        new_data = np.mean(elem, axis=0)
        data = np.vstack((data, new_data))
    return data

def augment_data_naive(x, y, factor):
    temp = np.asarray(y, dtype='bool')
    x_pos = augment_data_naive_helper(x[temp], factor)
    temp = np.invert(temp)
    x_neg = augment_data_naive_helper(x[temp], factor)
    x, y = shuffle(x_pos, x_neg, 1, 0)

    return x, y


def augment_data_smote(x, y, desired_sampels):
    ratio = {1:0.5*desired_samples, 0:0.5*desired_samples}
    x = flatten(x)
    new_x, new_y = SMOTE(ratio=ratio).fit_sample(x, y)
    x = make3D(x)
    new_x = make3D(new_x)
    x = np.vstack((x, new_x))
    y = np.concatenate((y, new_y))
    return x, y


def augment_data_adasyn(x, y, desired_sampels):
    ratio = {1:0.5*desired_sampels, 0:0.5*desired_sampels}
    x = flatten(x)
    new_x, new_y = ADASYN(ratio=ratio).fit_sample(x, y)
    x = make3D(x)
    new_x = make3D(new_x)
    x = np.vstack((x, new_x))
    y = np.concatenate((y, new_y))
    return x, y
