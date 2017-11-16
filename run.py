import models
import data
import argparse
import feature_scaling
import feature_selection
import data_augmentation
import numpy as np
import time


def run(model=models.support_vector_machine, model_args=None,
        data=data.get_kmer_us_uk_split, data_args=None,
        scaler=feature_scaling.scale_to_range, scaler_args=None,
        selection=None, selection_args=None,
        augment=None, augment_args=None, validate=True, record_time=False,
        record_std_dev=False, record_data_size=False, reps=10):
    """
    Parameters:
        model:          The machine learning model to be used, see
                        best_models.py.
        model_args:     The arguments to be passed to the model method.
        data:           The method used to gather the data, see data.py
        data_args:      The arguments to be passed to the data method
        scaler:         The method used to scale the data see
                        feature_scaling.py.
        scaler_args:    The arguments to be passed to eh scaler method.
        selection:      The method used to perform feature selection, see
                        feature_selection.py.
        selection_args: The arguments to be passed to the selection method.
        augment:        The method used to augment the training data, see
                        data_augmentation.py
        augment_args:   The arguments to be passed to the augment method.
        validate:       If true "data" should return x_train, y_train,
                        x_test, and y_test and "model" should accept the
                        output of data and return an accuracy. If false
                        "data" should return x_train, y_train, and x_test
                        and "model" should accept the output of "data" and
                        return predictions for x_test.
        reps:           How many times to run the model, if doing validation
    Returns:
        The output of "model" when given "data". If validating the model
        the output is the average over all repetitions.
    """
    if validate:
        results = np.zeros(reps)
        times = np.zeros(reps)
        for i in range(reps):
            if record_time:
                start = time.time()
            if data_args:
                d = data(*data_args)
            else:
                d = data()
            if selection:
                if selection_args:
                    d = selection(d, *selection_args)
                else:
                    d = selection(d)
            if d[0].max() > 1:
                if scaler_args:
                    d = scaler(d, *scaler_args)
                else:
                    d = scaler(d)
            if augment:
                if augment_args:
                    d = augment(d, *augment_args)
                else:
                    d = augment(d)
            if model_args:
                score = model(d, *model_args)
            else:
                score = model(d)
            results[i] = score
            if record_time:
                times[i] = time.time() - start
        if record_time:
            if record_std_dev:
                output = [results.mean(), results.std(), times.mean(), times.std()]
            else:
                output = [results.mean(), times.mean()]
        else:
            if record_std_dev:
                output = [results.mean(), results.std()]
            else:
                output = results.mean()
    else:
        if record_time:
            start = time.time()
        if not data_args:
            d = data()
        else:
            d = data(*data_args)
        if model_args:
            predictions = model(d, *model_args)
        else:
            predictions = model(d)
        if record_time:
            total_time = time.time() - start
            output = [prediction, total_time]
        else:
            output = predictions
    if record_data_size:
        if type(output) == list:
            output.append(d[0].shape[0]+d[2].shape[0])
        else:
            output = list(output).append(d[0].shape[0+d[2].shape[0]])
    return output

def model_methods(input_string):
    """
    Given a string that is the name of a model, return the model.
    """
    methods = models.get_methods()
    try:
        output = methods[input_string]
    except KeyError as E:
        print "%s is not a valid model name, using default model" % E
        output = None
    return output


def data_methods(input_string):
    """
    Given a string that is the name of data method, return the data method.
    """
    methods = data.get_methods()
    try:
        output = methods[input_string]
    except KeyError as E:
        print "%s is not a valid data method, using the default" % E
        output = None
    return output


def scale_methods(input_string):
    """
    Given a string that is the name of a scaling method, return the scaling
    method.
    """
    methods = feature_scaling.get_methods()
    try:
        output = methods[input_string]
    except KeyError as E:
        print "%s is not a valid scaling method, using the default" % E
        output = None
    return output


def selection_methods(input_string):
    """
    Given a string that is the name of a feature selection method, return
    the feature selection method.
    """
    methods = feature_selection.get_methods()
    try:
        output = methods[input_string]
    except KeyError as E:
        print "%s is not a valid selection method, using the default" % E
        output = None
    return output


def augment_methods(input_string):
    """
    Given a string that is the name of a data augmentation method, return
    the data augmentation method.
    """
    methods = data_augmentation.get_methods()
    try:
        output = methods[input_string]
    except KeyError as E:
        print "%s is not a valid augmentation method, using the default" % E
        output = None
    return output


def clean_args(value):
    """
    Convert strings to their proper type.
    """
    if value.isdigit():
        output = int(value)
    elif value == 'True':
        output = True
    elif value == 'False':
        output = False
    else:
        output = value
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        help='Machine learning model, see best_models.py',
                        type=model_methods)
    parser.add_argument('--model_arguments', '-ma',
                        help='The arguments to be passed to the model method.',
                        nargs='*',
                        type=clean_args)
    parser.add_argument('--data', '-d',
                        help='The method to get the input data, see data.py',
                        type=data_methods)
    parser.add_argument('--data_args', '-da',
                        help='The arguments to be passed to the data method',
                        nargs='*',
                        type=clean_args)
    parser.add_argument('--scaler', '-S',
                        help='The scaling method to apply to the data.',
                        type=scale_methods)
    parser.add_argument('--scaler_args', '-Sa',
                        help='The arguments for the scaling method.',
                        nargs='*',
                        type=clean_args)
    parser.add_argument('--selection', '-s',
                        help='The method used to perform feature selection',
                        type=selection_methods)
    parser.add_argument('--selection_args', '-sa',
                        help='The arguments for the feature selection method',
                        nargs='*',
                        type=clean_args)
    parser.add_argument('--augment', '-a',
                        help='The method used to augment the trianing data',
                        type=augment_methods)
    parser.add_argument('--augment_args', '-aa',
                        help='The arguments for the augment method.',
                        nargs='*',
                        type=clean_args)
    parser.add_argument('--reps', '-r',
                        help='How many times to run the model, ignored if validate is False',
                        type=int)
    parser.add_argument('--validate', '-v',
                        help='If True the model should return a score if False the model should return predictions',
                        type=clean_args,
                        choices=[True, False])
    parser.add_argument('--record_time', '-rt',
                        help='If true keep track of time, if flase do not',
                        type=clean_args,
                        choice=[True, False])
    parser.add_argument('--record_std_dev', '-rsd',
                        help='If true record the standard deviation or all results',
                        type=clean_args,
                        choice=[True, False])
    parser.add_argument('--record_data_size', 'rds',
                        help='If True output how many training and test samples there are',
                        type=clean_args,
                        choice=[True,False])
    args = parser.parse_args()
    args_dict = vars(args)
    # Remove unentered and invalid arguments
    filtered = {k: v for k,v in args_dict.items() if v is not None}
    output = run(**filtered)
    print output
