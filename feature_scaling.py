from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale_to_range(input_data, low=-1, high=1):
    """
    Scales the features in x_train and x_test to lie within the range low, high

    Args:
        input_data (tuple): x_train, y_train, x_test, y_test
        low (int):          The lower limit to scale the data to lie within.
        high (int):         The upper limit to scale the data to lie within.

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """
    x_train = np.asarray(input_data[0], dtype='float64')
    y_train = np.asarray(input_data[1])
    x_test = np.asarray(input_data[2], dtype='float64')
    y_test = np.asarray(input_data[3])

    scaler = MinMaxScaler(feature_range=(low, high))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return (x_train, y_train, x_test, y_test)
