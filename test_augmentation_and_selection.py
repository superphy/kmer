from US_UK_data import get_preprocessed_data
from data_augmentation import augment_data_naive, augment_data_smote, augment_data_adasyn
from feature_selection import variance_threshold, select_k_best
from feature_selection import select_percentile, recursive_feature_elimination
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv1D

feature_selectors = {'variance': [variance_threshold, 0.1],
        'k': [select_k_best, 100],
        'percentile': [select_percentile, 0.33],
        'recursive': [recursive_feature_elimination, 100]}

augmentation_methods = {'naive': [augment_data_naive, 2],
                       'smote': [augment_data_smote, 186],
                       'adasyn': [augment_data_adasyn, 186]}



def define_model(x_train):
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

    return model



def just_feature_selection():
    selectors = feature_selectors.keys()
    ans = {'default':0}
    maximum = 'default'
    x_train, y_train, x_test, y_test = get_preprocessed_data("database",True,False,7,13)
    for selector in selectors:
        method = feature_selectors[selector][0]
        param = feature_selectors[selector][1]
        x_train, y_train, x_test, y_test = method(x_train,y_train,x_test,y_test,param)
        model = define_model(x_train)
        model.fit(x_train, y_train, batch_size=10, epochs=120, verbose=1)
        score = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
        print selector, score[1]
        ans[selector] = score[1]
        if score[1] >= maximum:
            maximum = selector

    return maximum, ans[maximum]



def just_data_augmentation():
    augmentations = augmentation_methods.keys()
    ans = {'default':0}
    maximum = 'default'
    x_train, y_train, x_test, y_test = get_preprocessed_data("database",True,False,7,13)
    for augmentation in augmentations:
        method = augmentation_methods[augmentation][0]
        param = augmentation_methods[augmentaion][1]
        x_train, y_train = method(x_train,y_train,param)
        model = define_model(x_train)
        model.fit(x_train, y_train, batch_size=10, epochs=120, verbose=1)
        score = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
        print augmentation, score[1]
        ans[augmentation] = score[1]
        if score[1] >= ans[maximum]:
            maximum = augmentation

    return maximum, ans[maximum]



def both_slection_and_augmentation():
    augmentations = augmentation_methods.keys()
    selectors = feature_selectors.keys()
    ans = {'default':0}
    maximum = 'default'
    x_train, y_train, x_test, y_test = get_preprocessed_data("database",True,False,7,13)
    for augmentation in augmentations:
        for selector in selectors:
            aug_method = augmentation_methods[augmentation][0]
            aug_param = augmentation_methods[augmentaion][1]
            sel_method = feature_selectors[selector][0]
            sel_param = feature_selectors[selector][1]
            x_train,y_train,x_test,y_test = sel_method(x_train,y_train,x_test,y_test,sel_param)
            x_train, y_train = aug_method(x_train, y_train, aug_param)
            model = define_modle(x_train)
            model.fit(x_train, y_train, batch_size=10, epochs=120, verbose=1)
            score = model.evaluate(x_test, y_test, batch_size=10, verbsoe=0)
            key = augmentation + " " + selector
            print key, score[1]
            ans[key] = score[1]
            if score[1] >= ans[maximum]:
                maximum = key

    return maximum, ans[maximum]



def main():
    best = just_feature_selection()
    print "Just Feature Selection: ", best
    best = just_data_augmentation()
    print "Just Data Augmentation: ", best
    best = both_slection_and_augmentation()
    print "Both Feature Selection and Data Augmentation: ", best


if __name__ == "__main__":
    main()
