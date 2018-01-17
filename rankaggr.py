"""
A collection of methods that perform rank aggregation. Combining multiple
ranked lists into one overall ranking.
"""


import math
import numpy as np


def vote(all_results, voters=10, votes=10):
    """
    Performs rank aggregation by summing votes from each run, where each vote
    is equally weighted.

    Args:
        all_results:
        voters:
        votes:

    Returns:
        list:
    """
    weight = 1.0 / (voters * votes)
    parsed_results = []
    for result in all_results[:voters]:
        parsed_results.extend(result[:votes])
    parsed_results = np.asarray(parsed_results)
    parsed_results = parsed_results.reshape(parsed_results.shape[0])
    unique_votes, counts = np.unique(parsed_results, return_counts=True)
    counts = [x * weight for x in counts]
    output = dict(zip(unique_votes, counts))
    return output


def linear_decay(all_results, voters=10, votes=10):
    """
    Performs rank aggregation by applying linear decay to the ranked lists
    and summing those results. A final score of 1.0 is the best possible score.

    Args:
        all_results:
        base:
        voters:
        votes:

    Returns:
        list:
    """
    weight = 1.0 / (voters * votes)
    output = {}
    for result in all_results[:voters]:
        for elem in result[:votes]:
            if elem not in output:
                output[elem] = 0
            output[elem] += weight * (votes - result.index(elem))
    return output


def logarithmic_decay(all_results, base=2.0, voters=10, votes=10):
    """
    Performs rank aggregation by applying logarithmic decay to the ranked lists
    and summing those results. A final score of 1.0 is the best possible score.

    Args:
        all_results:
        base:
        voters:
        votes:

    Returns:
        list:
    """
    weight = 1.0 / voters
    output = {}
    for result in all_results[:voters]:
        for elem in result[:votes]:
            if elem not in output:
                output[elem] = 0
            output[elem] += weight * (1.0 / math.pow(base, result.index(elem)))
    return output


def exponential_decay(all_results, n_not=1.0, constant=1.0, voters=10,
                      votes=10):
    """
    Performs rank aggregation by applying exponential decay to the ranked lists
    and summing those results. A final score of 1.0 is the best possible score

    Args:
        all_results (list):
        n_not:
        constant:
        voters:
        votes:

    Returns:
        list:
    """
    weight = 1.0 / voters
    output = {}
    for result in all_results[:voters]:
        for elem in result[:votes]:
            if elem not in output:
                elem[output] = 0.0
            power = -constant * result.index(elem)
            elem[output] += (n_not * (math.pow(math.e, power))) * weight
    return output
