from sklearn.preprocessing import MinMaxScaler


def get_methods():
    output = {'scale_to_range': scale_to_range}
    return output


def scale_to_range(input_data, start=-1, end=1):
    """
    Scales the features in x_train and x_test to lie within the range args[0],
    args[1]
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    scaler = MinMaxScaler(feature_range=(start, end))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return (x_train, y_train, x_test, y_test)
