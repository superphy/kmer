import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.pooling import AveragePooling1D
from keras.layers.convolutional import Conv1D
from sklearn import svm

def neural_network(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape = (x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    evaluation = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
    return evaluation[1]

def nn(x_train, y_train, predict, binarize):
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape = (x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    prediction = model.predict(predict)
    if binarize:
        prediction = np.where(prediction > 0.5, 1, 0)
    return prediction

def support_vector_machine(x_train, y_train, x_test, y_test):
    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def linear_svm(x_train, y_train, predict):
    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model.predict(predict)
