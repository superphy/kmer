import best_models
import data
import argparse

def run(model=best_models.support_vector_machine, data=data.get_kmer_us_uk_split,
        reps=10, params=None, validate_model=True):
    if validate_model:
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
        print E, "Using default model"
        return None
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
        print E, "Using default data method"
        return None
    return output

def clean_params(input):
    """
    Convert strings to theif proper type.
    """
    if input.isdigit():
        return int(input)
    elif input == 'True':
        return True
    elif input == 'False':
        return False
    else:
        return input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',
                        help='Machine learning model to use, see best_models.py',
                        type=model_functions)
    parser.add_argument('-data',
                        help='The method to get the input data, see data.py',
                        type=data_functions)
    parser.add_argument('-reps',
                        help='How many times to run the model, ignored if validate_model is False',
                        type=int)
    parser.add_argument('-validate_model',
                        help='If True the model should return a score if False the model should return predictions',
                        type=bool)
    parser.add_argument('-params',
                        help='The parameters to be passed to the data_method',
                        nargs=argparse.REMAINDER,
                        type=clean_params)
    args = parser.parse_args()
    args_dict = vars(args)
    filtered = {k: v for k,v in args_dict.items() if v is not None}
    output = run(**filtered)
    print output
