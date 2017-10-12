from sklearn.model_selection import StratifiedShuffleSplit as ssSplit
from kmer_counter import count_kmers, get_counts
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import os
import sys



def __setup_files(filepath):
    """
    Takes a path to a directory, returns a list of the complete paths to each
    file in the directory
    """
    if not filepath[-1] == '/':
        filepath += '/'
    return [filepath + x for x in os.listdir(filepath)]



def __sensitivity_specificity(predicted_values, true_values):
    """
    Takes two arrays, one is the predicted_values from running a prediction, the other is
    the true values. Returns the sensitivity and the specificity of the machine
    learning model.
    """
    true_pos = len([x for x in true_values if x == 1])
    true_neg = len([x for x in true_values if x == 0])
    false_pos = 0
    false_neg = 0
    for i in range(len(predicted_values)):
        if true_values[i] == 0 and predicted_values[i] == 1:
            false_pos += 1
        if true_values[i] == 1 and predicted_values[i] == 0:
            false_neg += 1

    sensitivity = (1.0*true_pos)/(true_pos + false_neg)
    specificity = (1.0*true_neg)/(true_neg + false_pos)

    return sensitivity, specificity



def __make_predictions(train_data, train_labels, test_data, test_labels):
    """
    Takes training data and labels, and test data for an svm. If test_lables is
    None returns predictions on test_data, other wise returns a percentage of
    predictions that the svm got correct.
    """
    # stochastic = SGDClassifier(penalty='l1', loss='perceptron', alpha=0.01,
    #                         tol=float(1e-3), max_iter=1000)
    linear = svm.SVC(kernel='linear')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(train_data)
    Z = scaler.transform(test_data)

    try:
        linear.fit(X, train_labels)
        if test_labels:
            score = linear.score(Z, test_labels)
            return score
        else:
            return linear.predict(Z)
    except (ValueError, TypeError) as E:
        print E
        return -1



def run(k, limit, num_splits, pos, neg, predict):
    """
    Uses kmer counts of genomes as input to a support vector machine to predict
    presence or absence of a phenotype in a genome.

    Parameters:
        k:          The length of kmer to input
        l:          A lower limit on how many times a kmer needs to be present
                    in a genome in order to be ouput.
        num_splits: How many times to train and test the model. Used only if
                    predict is None.
        pos:        Path to a directory containing fasta files for genomes that
                    are positive for a phenotype.
        neg:        Path to a directory containing fasta files for genomes that
                    are negative for a phenotype.
        predict:    Path to a directory containing fasta files for genomes that
                    you want to perdict whether or not they have a phenotype.
                    Can be None, if None the files in pos and neg will be
                    shuffled and split so that 80%% are used to train the model
                    and 20%% to test a model.
    Returns:
        Predict is None:        A tuple with three values: percentage of
                                predictions correctly made, the sensitivity, and
                                the specificity. If num_splits greater than 1,
                                these are averages from all runs.
        Predict is not None:    Returns a binary array, where 1 means the
                                genome belongs to the pos group and 0 means the
                                genome belongs to the neg group.
    """

    if not predict:
        files = pos + neg
    else:
        files = pos + neg + predict

    count_kmers(k, limit, files, "database")

    arrays = get_counts(files, "database")

    labels=[1 for x in range(len(pos))]+[0 for x in range(len(neg))]

    if not predict:
        sss = ssSplit(n_splits=num_splits, test_size=0.2, random_state=42)

        score_total = 0.0

        for indices in sss.split(arrays, labels):
            X = [arrays[x] for x in indices[0]]
            Y = [labels[x] for x in indices[0]]
            Z = [arrays[x] for x in indices[1]]
            ZPrime = [labels[x] for x in indices[1]]

            score = __make_predictions(X,Y,Z,ZPrime)
            score_total += score

        output = score_total/num_splits

    else:
        sss = ssSplit(n_splits=1, test_size = 0.5, random_state=13)

        for indices in sss.split(arrays[:(len(pos)+len(neg))], labels):
            X = [arrays[x] for x in indices[0]]
            X.extend([arrays[x] for x in indices[1]])
            Y = [labels[x] for x in indices[0]]
            Y.extend([labels[x] for x in indices[1]])

        Z = arrays[len(pos) + len(neg):]
        output = __make_predictions(X, Y, Z, None)

    return output



def main():
    """
    Handles kmer_prediction.py being run from the command line.
    """
    if len(sys.argv) in [6,7]:
        k = int(sys.argv[1])
        l = int(sys.argv[2])
        reps = int(sys.argv[3])
        pos = __setup_files(sys.argv[4])
        neg = __setup_files(sys.argv[5])
        if len(sys.argv) == 6:
            output = run(k, l, reps, pos, neg, None)
        else:
            predict = __setup_files(sys.argv[6])
            output = run(k, l, reps, pos, neg, predict)
        print output
    else:
        output = """
        Error: Wrong number of arguments, requires at least 5, at most 6
        First Argument: kmer length
        Second Argument: Minimum kmer count required for a kmer to be output
        Third Argument: Number of times to repeat the training and testing of
                        the model, if greater than 1 returns the average of all
                        runs.
        Fourth Argument: Path to directory containing fasta files for genomes
                         positive for a phenotype.
        Fifth Argument: Path to a directory containing fasta files negative for
                        a phenotype.
        Sixth Argument: (Optional) Path to a directory containing fasta files
                        that you want to predict whether or not they are
                        positive for the phenotype. If supplied, all the files
                        contianed in the fourth and fifth parameter will be used
                        to train the model. If not supplied the files contained
                        in the fourth and fifth argument will be shuffled and
                        80%% will be used to train the model and the rest will
                        be used to test the model.

        Example: python %s 10 5 5 /home/data/pos/ /home/data/neg/
        Counts kmers with length 10 removing any that appear fewer than 5 times
        then performs the training and testing of the model 5 times using the
        fasta files contained in the positive and negative directories.
        """ % sys.argv[0]
        print output



if __name__ == "__main__":
    main()
