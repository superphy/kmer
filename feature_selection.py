from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif, RFE, RFECV
from sklearn.svm import SVC

def flatten(data):
    data = data.reshape(data.shape[0], data.shape[1])
    return data

def make3D(data):
    data = data.reshape(data.shape + (1,))
    return data

def variance_threshold(x_train, y_train, x_test, y_test, threshold=0.16):
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

    return x_train, y_train, x_test, y_test


def select_k_best(x_train, y_train, x_test, y_test, score_func=f_classif, k=500):
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

    return x_train, y_train, x_test, y_test


def select_percentile(x_train, y_train, x_test, y_test, score_func=f_classif,
                      percentile=0.5):
    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = SelectPercentile(score_func=score_func, percentile=percentile)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test


def recursive_feature_elimination(x_train, y_train, x_test, y_test,
                                  estimator=SVC(kernel='linear', C=1),
                                  n_features=None, step=1):
    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = RFE(estimator, n_features, step)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test

def recursive_feature_elimination_cv(x_train, y_train, x_test, y_test,
                                     estimator=SVC(kernel='linear', C=1),
                                     step=1, cv=3):
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

    return x_train, y_train, x_test, y_test
