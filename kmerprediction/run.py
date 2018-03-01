"""
This is essentially the main method.

Provides a wrapper to gather data, preprocess the data, perform feature
selection, build the model, train the model, and test/use the model.
Chains together methods from get_data.py, feature_selection.py,
feature_scaling, data_augmentation.py, and models.py.
"""

from builtins import zip
from builtins import range
import inspect
import time
import datetime
import argparse
from kmerprediction import models
from kmerprediction import get_data
from kmerprediction import feature_scaling
from kmerprediction import feature_selection
from kmerprediction import data_augmentation
import numpy as np
import yaml
from kmerprediction.utils import do_nothing


def run(model=models.support_vector_machine, model_args=None,
        data_method=get_data.get_kmer_us_uk_split, data_args=None,
        scaler=do_nothing, scaler_args=None, selection=do_nothing,
        selection_args=None, augment=do_nothing, augment_args=None,
        validate=False, reps=10, collect_features=True):
    """
    Chains a data gathering method, data preprocessing methods, and a machine
    learning model together. Stores the settings for all the methods and the
    output of the model in a dictionary.

    Args:
        model (function):       The machine learning model to be used, see
                                models.py.
        model_args (dict):      The arguments to be passed to the model.
        data_method (function): The method used to gather the data, see
                                get_data.py
        data_args (dict):       The arguments to be passed to the data method
        scaler (function):      The method used to scale the data see
                                feature_scaling.py.
        scaler_args (dict):     The arguments to be passed to the scaler method
        selection (function):   The method used to perform feature selection,
                                see feature_selection.py.
        selection_args (dict):  The arguments to be passed to the selection
                                method.
        augment (function):     The method used to augment the training data,
                                see data_augmentation.py
        augment_args (dict):    The arguments to be passed to the augment
                                method.
        validate (bool):        If true the model will return an accuracy. If
                                false the model will return a dictionary of
                                inputs and their predictions.
        reps (int):             How many times to run the model, used only when
                                doing validation otherwise set to 1.
        collect_features (bool):If true, a list of dictionaries containing
                                keys of feature names and values of their
                                feature importance from each run is returned.

    Returns:
        (dict):   Contains all of the arguments and results from the run.
    """
    scaler = scaler or do_nothing
    selection = selection or do_nothing
    augment = augment or do_nothing

    model_args = model_args or {}
    data_args = data_args or {}
    scaler_args = scaler_args or {}
    selection_args = selection_args or {}
    augment_args = augment_args or {}

    if validate:
        results = np.zeros(reps)
    else:
        reps = 1
    times = np.zeros(reps)
    train_sizes = np.zeros(reps)
    test_sizes = np.zeros(reps)

    data_args['validate'] = validate
    model_args['validate'] = validate

    feature_importances = []
    for i in range(reps):
        start = time.time()
        # Get input data
        data, features, files, le = data_method(**data_args)

        # Perform feature selection on input_data
        selection_args['feature_names'] = features
        data, features = selection(data, **selection_args)
        selection_args.pop('feature_names', None)

        # Scale input data
        data = scaler(data, **scaler_args)

        # Augment training data
        data = augment(data, **augment_args)

        # Build and use the model
        model_args['feature_names'] = features
        output_data, features = model(data, **model_args)
        model_args.pop('feature_names', None)

        # Record information about run
        times[i] = time.time() - start
        if validate:
            results[i] = output_data
        else:
            results = output_data
        train_sizes[i] = data[0].shape[0]
        test_sizes[i] = data[2].shape[0]
        feature_importances.append(features)

    # Store information about the run in a dictionary
    output = {}
    output['train_sizes'] = train_sizes.mean().tolist()
    output['test_sizes'] = test_sizes.mean().tolist()
    output['avg_run_time'] = times.mean().tolist()
    output['std_dev_run_times'] = times.std().tolist()
    output['num_genomes'] = data[0].shape[0] + data[2].shape[0]

    if validate:
        # Compute the mean and std dev of all the runs
        output['avg_result'] = results.mean().tolist()
        output['std_dev_results'] = results.std().tolist()
        output['results'] = results.tolist()
    else:
        # Create dictionary with test files as keys and predictions as values
        # Convert classes back to their original values
        results = le.inverse_transform(results)
        output['results'] = dict(list(zip(files, results.tolist())))

    if collect_features:
        output['important_features'] = feature_importances

    output['repetitions'] = reps
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
    output['datetime'] = datetime.datetime.now()

    all_labels = np.concatenate((data[1], data[3]))
    classes, class_counts = np.unique(all_labels, return_counts=True)
    classes = le.inverse_transform(classes).tolist()
    class_counts = class_counts.tolist()
    class_sample_sizes = dict(zip(classes, class_counts))
    output['class_sample_sizes'] = class_sample_sizes

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
    for x in [models, get_data, feature_selection,
              feature_scaling, data_augmentation]:
        temp = dict(inspect.getmembers(x, inspect.isfunction))
        methods.update(temp)
    return methods


def convert_methods(input_dictionary):
    """
    Converts any method names in the input dictionary to actual methods.

    Args:
        input_dictionary (dict): A dictionary of kwargs for run.

    Returns:
        (dict): Same keys as input_dictionary, values that are method names are
                converted to actual methods.
    """
    output_dictionary = {}
    methods = get_methods()
    for key, value in list(input_dictionary.items()):
        if isinstance(value, dict):
            output = convert_methods(value)
        elif value in list(methods.keys()):
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


def main(input_yaml, output_yaml, name):
    """
    Reads config from a yaml file, performs the run, and saves the results and
    configuration in another yaml file.

    Args:
        input_yaml (str):   Filepath to a yaml file containing the run config.
        output_yaml (str):  Filepath to a yaml file where the results will be
                            stored.
        name (str):         What the yaml document will be named in output_yaml

    Returns:
        None
    """
    args = convert_yaml(input_yaml)
    run_output = run(**args)
    document = {'name': name, 'output': run_output}
    with open(output_yaml, 'a') as output_file:
        yaml.dump(document, output_file, explicit_start=True,
                  explicit_end=True, default_flow_style=False,
                  allow_unicode=True)
        output_file.write('\n\n\n')


if __name__ == "__main__":
    cl_args = create_arg_parser()
    main(cl_args.input, cl_args.output, cl_args.name)
