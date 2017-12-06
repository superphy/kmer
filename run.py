import models
import data
import sys
import feature_scaling
import feature_selection
import data_augmentation
import inspect
import numpy as np
import time
import datetime
import yaml
import constants
from sklearn.feature_selection import chi2, f_classif


methods = {}
for x in [models,data,feature_selection,feature_scaling,data_augmentation]:
    temp = dict(inspect.getmembers(x))
    temp = {k:v for k,v in temp.items() if inspect.isfunction(v)}
    methods.update(temp)


def run(model=models.support_vector_machine, model_args=None,
        data=data.get_kmer_us_uk_split, data_args=None,
        scaler=feature_scaling.scale_to_range, scaler_args=None,
        selection=None, selection_args=None,
        augment=None, augment_args=None, validate=False, reps=10):
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
    output = {}
    output['datetime'] = datetime.datetime.now()
    if validate:
        results = np.zeros(reps)
        times = np.zeros(reps)
        train_sizes = np.zeros(reps)
        test_sizes = np.zeros(reps)
        for i in range(reps):
            start = time.time()
            if data_args:
                d = data(**data_args)
            else:
                d = data()
            if selection:
                if selection_args:
                    d = selection(d, **selection_args)
                else:
                    d = selection(d)
            if scaler:
                if scaler_args:
                    d = scaler(d, **scaler_args)
                else:
                    d = scaler(d)
            if augment:
                if augment_args:
                    d = augment(d, **augment_args)
                else:
                    d = augment(d)
            if model_args:
                score = model(d, **model_args)
            else:
                score = model(d)
            times[i] = time.time() - start
            results[i] = score
            train_sizes[i] = d[0].shape[0]
            test_sizes[i] = d[2].shape[0]
        output['train_sizes'] = train_sizes.tolist()
        output['test_sizes'] = test_sizes.tolist()
        output['avg_run_time'] = times.mean().tolist()
        output['std_dev_run_times'] = times.std().tolist()
        output['avg_result'] = results.mean().tolist()
        output['std_dev_results'] = results.std().tolist()
        output['results'] = results.tolist()
        output['run_times'] = times.tolist()
        output['repetitions'] = reps
    else:
        start = time.time()
        if data_args:
            d = data(**data_args)
        else:
            d = data()
        if selection:
            if selection_args:
                d = selection(d, **selection_args)
            else:
                d = selection(d)
        if scaler:
            if scaler_args:
                d = scaler(d, **scaler_args)
            else:
                d = scaler(d)
        if augment:
            if augment_args:
                d = augment(d, **augment_args)
            else:
                d = augment(d)
        if model_args:
            predictions = model(d, **model_args)
        else:
            predictions = model(d)
        total_time = time.time() - start
        output['predictions'] = predictions.tolist()
        output['run_time'] = total_time
        output['train_size'] = len(d[0])
        output['test_size'] = len(d[2])
    output['model'] = model
    output['model_args'] = model_args
    output['data'] = data
    output['data_args'] = data_args
    output['scaler'] = scaler
    output['scaler_args'] = scaler_args
    output['selection'] = selection
    output['selection_args'] = selection_args
    output['augment'] = augment
    output['augment_args'] = augment_args
    return output

def parse_args(input_dictionary):
    output_dictionary = {}
    for key, value in input_dictionary.items():
        if type(value) == dict:
            output = parse_args(value)
        elif value in methods.keys():
            output = methods[value]
        else:
            output = value
        output_dictionary[key] = output
    return output_dictionary

def convert_yaml(input_file):
    output_dictionary = {}
    with open(input_file, 'r') as f:
        input_dictionary = yaml.load(f)
    output_dictionary = parse_args(input_dictionary)
    return output_dictionary

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        config_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        config_file = sys.argv[1]
        output_file = constants.OUTPUT
    else:
        config_file = constants.CONFIG
        output_file = constants.OUTPUT
    args = convert_yaml(config_file)
    output = run(**args)
    with open(output_file, 'a') as f:
        f.write('---\n')
        f.write('#%s\n'%str(datetime.datetime.now()))
        yaml.dump(output, f)
        f.write('...\n')
