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
import argparse
from sklearn.feature_selection import chi2, f_classif
import utils


def run(model=models.support_vector_machine, model_args={},
        data=data.get_kmer_us_uk_split, data_args={},
        scaler=feature_scaling.scale_to_range, scaler_args={}, selection=None,
        selection_args={}, augment=None, augment_args={}, validate=False,
        reps=10, extract=False):
    """
    Chains a data gathering method, data preprocessing methods, and a machine
    learning model together. Stores the settings for all the methods and the
    output of the model in a dictionary.

    Args:
        model (function):       The machine learning model to be used, see
                                best_models.py.
        model_args (dict):      The arguments to be passed to the model method.
        data (function):        The method used to gather the data, see data.py
        data_args (dict):       The arguments to be passed to the data method
        scaler (function):      The method used to scale the data see
                                feature_scaling.py.
        scaler_args (dict):     The arguments to be passed to eh scaler method.
        selection (function):   The method used to perform feature selection, see
                                feature_selection.py.
        selection_args (dict):  The arguments to be passed to the selection method.
        augment (function):     The method used to augment the training data, see
                                data_augmentation.py
        augment_args (dict):    The arguments to be passed to the augment method.
        validate (bool):        If true "data" should return x_train, y_train,
                                x_test, and y_test and "model" should accept the
                                output of data and return an accuracy. If false
                                "data" should return x_train, y_train, and x_test
                                and "model" should accept the output of "data" and
                                return predictions for x_test.
        reps (int):             How many times to run the model, if doing validation
        extract (bool):         If true the most important feautres used to
                                to do the classification are returned.

    Returns:
        A dictionary containing all of the passed arguments and the results of
        the run.
    """
    output = {}
    output['datetime'] = datetime.datetime.now()
    if validate:
        results = np.zeros(reps)
        times = np.zeros(reps)
        train_sizes = np.zeros(reps)
        test_sizes = np.zeros(reps)
        if extract:
            features = []
        for i in range(reps):
            start = time.time()
            if extract:
                data_args['extract'] = True
                d,f = data(**data_args)
            else:
                d = data(**data_args)
            output['num_genomes'] = d[0].shape[0] + d[2].shape[0]
            if selection:
                if extract:
                    selection_args['feature_names']=f
                    d,f = selection(d, **selection_args)
                    selection_args.pop('feature_names', None)
                else:
                    d = selection(d, **selection_args)
            if scaler:
                d = scaler(d, **scaler_args)
            if augment:
                d = augment(d, **augment_args)
            if extract:
                model_args['feature_names'] = f
                score, f = model(d, **model_args)
                model_args.pop('feature_names', None)
            else:
                score = model(d, **model_args)
            times[i] = time.time() - start
            results[i] = score
            train_sizes[i] = d[0].shape[0]
            test_sizes[i] = d[2].shape[0]
            if extract:
                features.append(f)
        output['train_sizes'] = train_sizes.mean().tolist()
        output['test_sizes'] = test_sizes.mean().tolist()
        output['avg_run_time'] = times.mean().tolist()
        output['std_dev_run_times'] = times.std().tolist()
        output['avg_result'] = results.mean().tolist()
        output['std_dev_results'] = results.std().tolist()
        output['results'] = results.tolist()
        output['repetitions'] = reps
        if extract:
            utils.make_unique(features)
            output['important_features'] = features
    else:
        start = time.time()
        if extract:
            data_args['extract'] = True
            d,f = data(**data_args)
        else:
            d = data(**data_args)
        output['num_genomes'] = d[0].shape[0] + d[2].shape[0]
        if selection:
            if extract:
                selection_args['feature_names'] = f
                d,f = selection(d, **selection_args)
                selection_args.pop('feature_names', None)
            else:
                d = selection(d, **selection_args)
        if scaler:
            d = scaler(d, **scaler_args)
        if augment:
            d = augment(d, **augment_args)
        if extract:
            model_args['feature_names'] = f
            predictions, f = model(d, **model_args)
            model_args.pop('feature_names', None)
        else:
            predictions = model(d, **model_args)
        total_time = time.time() - start
        output['predictions'] = predictions.tolist()
        output['run_time'] = total_time
        output['train_size'] = len(d[0])
        output['test_size'] = len(d[2])
        if extract:
            f = [utils.convert_well_index(x) for x in f]
            output['important_features'] = f.tolist()
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


def get_methods():
    """
    Gets a dictionary of all the methods defined and imported in the files
    models.py, data.py, feature_selection.py feature_scaling.py, and
    data_augmentation.py

    Args:
        None

    Returns:
        dict(str:function)
    """
    methods = {}
    for x in [models,data,feature_selection,feature_scaling,data_augmentation]:
        temp = dict(inspect.getmembers(x, inspect.isfunction))
        methods.update(temp)
    return methods


def convert_methods(input_dictionary):
    """
    Converts any method names that are values in the input dictionary to actual
    methods.

    Args:
        input_dictionary (dict): A dictionary of kwargs for run.

    Returns:
        (dict): Same keys as input_dictionary, values that are method names are
                converted to actual methods.
    """
    output_dictionary = {}
    methods = get_methods()
    for key, value in input_dictionary.items():
        if type(value) == dict:
            output = convert_methods(value)
        elif value in methods.keys():
            output = methods[value]
        else:
            output = value
        output_dictionary[key] = output
    return output_dictionary


def convert_yaml(input_file):
    """
    Converts the data in the input yaml file to a dictionary and passes the
    result through convert_methods.

    Args:
        input_file (str): Path to a yaml file containing the config for run.

    Returns:
        output (dict): The data contained in input_file converted to a
                       dictionary and passed through convert_methods.
    """
    output = {}
    with open(input_file, 'r') as f:
        input_dictionary = yaml.load(f)
    output = convert_methods(input_dictionary)
    return output

def create_arg_parser():
    """
    Creates a namespace object for the command line arguments of run.

    Args:
        None
    Returns:
        Namespace object populated with the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="""yaml configuration file for run. If not
                                provided Data/config.yml is used.""",
                        default='Data/config.yml')
    parser.add_argument("-o", "--output",
                        help="""yaml file where the results of the run will be
                                stored. If not provided Data/run_results.yml is
                                used.""",
                        default='Data/run_results.yml')
    parser.add_argument("-n", "--name",
                        help="""What the yaml document will be named in the
                                output file. If not provided the current
                                Datetime is used. If using spaces, surround
                                with quotes.""",
                        default=datetime.datetime.now())
    return parser.parse_args()


if __name__ == "__main__":
    cl_args = create_arg_parser()
    args = convert_yaml(cl_args.input)
    output = run(**args)
    document = {'name':cl_args.name, 'output':output}
    with open(cl_args.output, 'a') as f:
        yaml.dump(document, f, explicit_start=True, explicit_end=True,
                  default_flow_style=False, allow_unicode=True)
        f.write('\n\n\n')
