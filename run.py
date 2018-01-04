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
from utils import do_nothing


def run(model=models.support_vector_machine, model_args={},
        data_method=data.get_kmer_us_uk_split, data_args={}, scaler=do_nothing,
        scaler_args={}, selection=do_nothing, selection_args={},
        augment=do_nothing, augment_args={}, validate=False, reps=10):
    """
    Chains a data gathering method, data preprocessing methods, and a machine
    learning model together. Stores the settings for all the methods and the
    output of the model in a dictionary.

    Args:
        model (function):       The machine learning model to be used, see
                                best_models.py.
        model_args (dict):      The arguments to be passed to the model method.
        data_method (function): The method used to gather the data, see data.py
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

    Returns:
        (dict):   Contains all of the arguments and results from the run.
    """
    # Ensure that all stages of the run are all doing or not doing validation
    data_args['kwargs']['validate'] = validate

    # Ensure that non validation runs are done only once
    if not validate:
        reps = 1

    output = {}
    output['datetime'] = datetime.datetime.now()

    # convert optional methods to do_nothing if they are given as False or None
    scaler = scaler if scaler else do_nothing
    selection = selection if selection else do_nothing
    augment = augment if augment else do_nothing

    if validate:
        results = np.zeros(reps)
    times = np.zeros(reps)
    train_sizes = np.zeros(reps)
    test_sizes = np.zeros(reps)
    all_features = []

    for i in range(reps):
        start = time.time()
        # Get input data
        data,features,files,le = data_method(**data_args)
        output['num_genomes'] = data[0].shape[0] + data[2].shape[0]

        # Perform feature selection on input_data
        selection_args['feature_names']=features
        data,features = selection(data, **selection_args)
        selection_args.pop('feature_names', None)

        # Scale input data
        data = scaler(data, **scaler_args)

        # Augment training data
        data = augment(data, **augment_args)

        # Build and use the model
        model_args['feature_names'] = features
        output_data,features = model(data, **model_args)
        model_args.pop('feature_names', None)

        # Record information about run
        times[i] = time.time() - start
        if validate:
            results[i] = output_data
        else:
            results = output_data
        train_sizes[i] = data[0].shape[0]
        test_sizes[i] = data[2].shape[0]
        all_features.append(features)

    # Store information about the run in a dictionary
    output['train_sizes'] = train_sizes.mean().tolist()
    output['test_sizes'] = test_sizes.mean().tolist()
    output['avg_run_time'] = times.mean().tolist()
    output['std_dev_run_times'] = times.std().tolist()

    if validate:
        # Compute the mean and std dev of all the runs
        output['avg_result'] = results.mean().tolist()
        output['std_dev_results'] = results.std().tolist()
        output['results'] = results.tolist()
    else:
        # Create dictionary with test files as keys and their predictions as values
        results = le.inverse_transform(results) # Convert classes back to their orignal values
        import pdb; pdb.set_trace()
        output['results'] = dict(zip(files, results.tolist()))
    output['repetitions'] = reps

    all_features = list(np.concatenate(all_features, axis=0))
    feature_counts = dict()
    for f in all_features:
        feature_counts[str(f)]=feature_counts.get(f,0)+1
    feature_counts = {utils.convert_well_index(k):v for k,v in feature_counts.items()}

    output['important_features'] = feature_counts
    output['model'] = model
    output['model_args'] = model_args
    output['data'] = data_method
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
