from sklearn.preprocessing import MinMaxScaler


def get_methods():
    output = {'scale_to_range': scale_to_range}
    return output


def scale_to_range(x_train, y_train, x_test, y_test, args=[-1, 1]):
    """
    Scales the features in x_train and x_test to lie within the range args[0],
    args[1]
    """
    low = args[0]
    high = args[1]
    scaler = MinMaxScaler(feature_range=(low, high))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, y_train, x_test, y_test
