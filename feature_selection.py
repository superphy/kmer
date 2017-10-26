from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.svm import SVR

def flatten(data):
    data = data.reshape(data.shape[0], data.shape[1])
    return data

def make3D(data):
    data = data.reshape(data.shape + (1,))
    return data

def variance_threshold(x_train, y_train, x_test, y_test, threshold):
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    feature_selector = VarianceThreshold(threshold=threshold)
    x_train = feature_selector.fit_transform(x_train)
    x_test = feature_selector.transform(x_test)
    x_train = make3D(x_train)
    x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test


def select_k_best(x_train, y_train, x_test, y_test, k):
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    feature_selector = SelectKBest(f_classif, k=k)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    x_train = make3D(x_train)
    x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test


def select_percentile(x_train, y_train, x_test, y_test, percentile):
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    feature_selector = SelectPercentile(f_classif, percentile=percentile)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    x_train = make3D(x_train)
    x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test

def recursive_feature_elimination(x_train, y_train, x_test, y_test, n_features):
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    estimator = SVR(kernel='linear')
    feature_selector = RFE(estimator, n_features, step=1)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.fit(x_test, y_test)
    x_train = make3D(x_train)
    x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test
