from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif, RFE, RFECV
from sklearn.svm import SVC
from utils import flatten, make3D

def get_methods():
    output = {"variance": variance_threshold,
              "k_best": select_k_best,
              "percentile": select_percentile,
              "rfe": recursive_feature_elimination,
              "rfecv": recursive_feature_elimination_cv}
    return output

def variance_threshold(x_train, y_train, x_test, y_test, args=[0.16]):
    """
    Removes all features from x_train and x_test whose variances is less than
    args[0] in x_train.
    """
    threshold = args[0]
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


def select_k_best(x_train, y_train, x_test, y_test, args=[f_classif, 500]):
    """
    Selects the args[1] best features in x_train, removes all others from
    x_train and x_test. Selects the best features by using the score function
    args[0].
    """
    score_func = args[0]
    k = args[1]
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


def select_percentile(x_train, y_train, x_test, y_test, args=[f_classif, 0.5]):
    """
    Selects the best args[1] percentile of features in x_train, removes the rest
    of the features from x_train and x_test. Selects the best features by using
    the score function args[0].
    """
    score_func = args[0]
    percentile = args[1]
    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = SelectPercentile(score_func=score_func,
                                        percentile=percentile)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test


def recursive_feature_elimination(x_train, y_train, x_test, y_test,
                                  args=[SVC(kernel='linear', C=1), None, 1]):
    """
    Recursively eliminates features from x_train and x_test, see documentaion:
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    for more details.
    If providing args, it should be of the form:
    [estimator, n_features_to_select, step]
    """
    estimator = args[0]
    n_features_to_select = args[1]
    step = args[2]
    dims = len(x_train.shape)
    if dims == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    feature_selector = RFE(estimator, n_features_to_select, step)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_test = feature_selector.transform(x_test)
    if dims == 3:
        x_train = make3D(x_train)
        x_test = make3D(x_test)

    return x_train, y_train, x_test, y_test

def recursive_feature_elimination_cv(x_train, y_train, x_test, y_test,
                                     args=[SVC(kernel='linear', C=1), 1, 3]):
    """
    Recursively elinates features from x_traina nd x_test using cross validation
    see documentation:
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    for more details.
    If providing args, it should be of the form: [estimator, step, cv]
    """
    estimator = args[0]
    step = args[1]
    cv = args[2]
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
