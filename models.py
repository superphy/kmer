import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.pooling import AveragePooling1D
from keras.layers.convolutional import Conv1D
from sklearn import svm
from utils import flatten, make3D, convert_to_numerical_classes
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier


def neural_network_validation(input_data):
    """
    Constructs, compiles, trains, and tests a neural network.
    Returns the accuracy of the model.
    """
    input_data, le = convert_to_numerical_classes(input_data)

    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]

    train_classes = np.unique(np.asarray(y_train))
    test_classes = np.unique(np.asarray(y_test))
    classes = np.unique(np.hstack((train_classes, test_classes)))
    num_classes = classes.shape[0]

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    if len(x_train.shape) == 2:
        x_train = make3D(x_train)
        x_test = make3D(x_test)
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     padding='same',
                     input_shape = (x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    evaluation = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    return evaluation[1]


def neural_network(input_data, binarize=True):
    """
    Constructs, compiles, trains and makes predictions with a neural network.
    Returns the predicted values for "predict". If args[0] is true the
    predictions will be 0's and 1's if not the predictions will be floats
    between 0.0 and 1.0 with values closer to 0.0 and 1.0 indicating a higher
    probability of the prediction being correct.
    If args is not provided, default values will be used.
    """
    input_data = input_data[:-1]
    input_data, le = convert_to_numerical_classes(input_data)

    print le.inverse_transform([0,1,2,3,4])

    x_train = input_data[0]
    y_train = input_data[1]
    predict = input_data[2]

    train_classes = np.unique(np.asarray(y_train))
    num_classes = len(list(train_classes))

    y_train = to_categorical(y_train)

    if len(x_train.shape) == 2:
        x_train = make3D(x_train)
        predict = make3D(predict)
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape = (x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    prediction = model.predict(predict)
    if binarize:
        prediction = np.where(prediction > 0.5, 1, 0)
    return prediction


def support_vector_machine_validation(input_data, kernel='linear', C=1,
                                      feature_names=None, num_features=10):
    """
    Fits, trains, and tests a support vector machine.
    Returns the accuracy of the model.
    """
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]
    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(x_train, y_train)
    output_data = model.score(x_test, y_test)

    if feature_names is not None:
        coefs = np.argsort(model.coef_.ravel())
        top_pos_features = feature_names[coefs[-num_features:]]
        top_neg_features = feature_names[coefs[:num_features]]
        top_features = np.hstack((top_pos_features,top_neg_features))
        output = (output_data, top_features)
    else:
        output = output_data

    return output


def support_vector_machine(input_data, kernel='linear', C=1, feature_names=None,
                           num_features=10):
    """
    Fits, trains, and makes predictions with a support vector machine.
    Returns the predicted values for "predict"
    """
    x_train = input_data[0]
    y_train = input_data[1]
    predict = input_data[2]

    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(x_train, y_train)
    output_data = model.predict(predict)

    if feature_names is not None:
        coefs = np.argsort(model.coef_.ravel())
        top_pos_features = feature_names[coefs[-num_features/2:]]
        top_neg_features = feature_names[coefs[:num_features/2]]
        output = (output_data, top_pos_features, top_neg_features)
    else:
        output = output_data

    return output

#First Optimization
# def random_forest_validation(input_data,n_estimators=5,criterion='gini',
#                              max_features=None,max_depth=None,
#                              min_samples_split=5,min_samples_leaf=1,
#                              min_weight_fraction_leaf=0.1,
#                              max_leaf_nodes=10,min_impurity_decrease=0,
#                              bootstrap=True, n_jobs=-1):
#Second Optimization
def random_forest_validation(input_data,n_estimators=50,criterion='entropy',
                             max_features='log2',max_depth=100,
                             min_samples_split=2,min_samples_leaf=1,
                             min_weight_fraction_leaf=0.01,
                             max_leaf_nodes=25,min_impurity_decrease=0.001,
                             bootstrap=False, n_jobs=-1, feature_names=None,
                             num_features=10):
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]
    if len(x_train.shape) == 3:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
    kwargs = {'n_estimators': n_estimators, 'criterion': criterion,
              'max_features': max_features, 'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf':min_samples_leaf,
              'min_weight_fraction_leaf': min_weight_fraction_leaf,
              'max_leaf_nodes':max_leaf_nodes,
              'min_impurity_decrease': min_impurity_decrease,
              'bootstrap': bootstrap, 'n_jobs': n_jobs}
    model = RandomForestClassifier(**kwargs)
    model.fit(x_train, y_train)
    output_data = model.score(x_test, y_test)

    if feature_names is not None:
        importances = np.argsort(model.feature_importances_.ravel())
        top_features = feature_names[importances[-num_features:]]
        output = (output_data, top_features)
    else:
        output = output_data
    return output


# The parameters in the below functions were found by performing a grid search.
# The resutls from the grid search are in ~/Data/sgdc_parameters/
def kmer_split_sgd(input_data):
    model = SGDC(loss='log', n_jobs=-1, eta0=1.0,
                 learning_rate='invscaling', penalty='none', tol=0.001,
                 alpha=100000000.0)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def kmer_mixed_sgd(input_data):
    model = SGDC(loss='squared_hinge', n_jobs=-1, penalty='none',
                 tol=0.001, alpha=10000000.0)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def genome_split_sgd(input_data):
    model = SGDC(loss='hinge', n_jobs=-1, eta0=0.1,
                 learning_rate='invscaling', penalty='l1', tol=0.001,
                 alpha=0.01)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def genome_mixed_sgd(input_data):
    model = SGDC(loss='log', n_jobs=-1, eta0=0.1,
                 learning_rate='invscaling', penalty='l1', tol=0.001,
                 alpha=0.001)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])
