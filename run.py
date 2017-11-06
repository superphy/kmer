import best_models
import data
import argparse

def run(model=best_models.support_vector_machine, data=data.get_kmer_us_uk_split,
        reps=10, params=None, validate=True):
    """
    Parameters:
        model:  The machine learning model to be used, see best_models.py
        data:   The method used to prepare the data for the model, see data.py
        reps:   How many times to run the model, if doing validation
        params: A list or tuple of all parameters to be passed to "data"
        validate_model: If true "data" should return x_train, y_train, x_test
                        and y_test and "model" should accept the output of data
                        and return an accuracy. If false "data" should return
                        x_train, y_train, and x_test and "model" should accept
                        the output of "data" and return predictions for x_test.
    Returns:
        The output of "model" when given "data". If validating the model the
        output is the average over all repetitions.
    """
    print model, data, reps, params, validate
    exit()
    if validate:
        total = 0.0
        for i in range(reps):
            if not params:
                d = data_method()
            else:
                d = data_method(*params)
            score = model(*d)
            total += score
        return total/reps
    else:
        if not params:
            d = data_method()
        else:
            d = data_method(*params)
        predictions = model(*d)
        return predictions

def model_functions(input):
    """
    Given a string that is the name of a model, return the model.
    """
    model_map = {'neural_network': best_models.neural_network,
                 'support_vector_machine': best_models.support_vector_machine,
                 'nn': best_models.nn,
                 'linear_svm': best_models.linear_svm}
    try:
        output = model_map[input]
    except KeyError as E:
        print "%s is not a valid model name, using default model" % E
        output = None
    return output

def data_functions(input):
    """
    Given a string that is the name of data method, return the data method.
    """
    data_map = {'get_kmer_us_uk_split': data.get_kmer_us_uk_split,
                'get_kmer_us_uk_mixed': data.get_kmer_us_uk_mixed,
                'get_genome_region_us_uk_mixed': data.get_genome_region_us_uk_mixed,
                'get_genome_region_us_uk_split': data.get_genome_region_us_uk_split,
                'get_kmer_data_from_json': data.get_kmer_data_from_json,
                'get_kmer_from_directory': data.get_kmer_from_directory}
    try:
        output = data_map[input]
    except KeyError as E:
        print "%s is not a valid data method, using default data method" % E
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
                        help='Machine learning model to use, see best_models.py',
                        type=model_functions)
    parser.add_argument('--data', '-d',
                        help='The method to get the input data, see data.py',
                        type=data_functions)
    parser.add_argument('--reps', '-r',
                        help='How many times to run the model, ignored if validate is False',
                        type=int)
    parser.add_argument('--validate', '-v',
                        help='If True the model should return a score if False the model should return predictions',
                        type=clean_args,
                        choices=[True, False])
    parser.add_argument('--params', '-p',
                        help='The parameters to be passed to the data method',
                        nargs=argparse.REMAINDER,
                        type=clean_args)
    args = parser.parse_args()
    args_dict = vars(args)
    filtered = {k: v for k,v in args_dict.items() if v is not None}
    print filtered
    output = run(**filtered)
    print output
