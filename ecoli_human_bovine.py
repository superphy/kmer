from kmer_prediction import run
from utils import get_human_path, get_bovine_path
import os
import sys

def setup_files():
    human_path = get_human_path()
    bovine_path = get_bovine_path()
    h_files = os.listdir(human_path)
    h_files = [human_path + x for x in h_files]

    b_files = os.listdir(bovine_path)
    b_files = [bovine_path + x for x in b_files]

    return h_files, b_files

def main():
    if len(sys.argv) == 4:
        k = int(sys.argv[1])
        l = int(sys.argv[2])
        rep = int(sys.argv[3])
        pos, neg = setup_files()
        output = run(k, l, rep, pos, neg, None)
        print output
    else:
        output = """
        Error: Wrong number of a rguments, requires exactly 3
        First Argument: kmer length
        Second Argument: Minimum kmer count required for a kmer to be output
        Third Argument: Number of times to repeat the training and testing of
                        the model, if greater than 1 returns the average of all
                        runs.

        Example: python %s 10 5 5
        Counts kmers with length 10 removing any that appear fewer than 5 times
        then performs the training and testing of the model 5 times.
        """ % sys.argv[0]
        print output

if __name__ == "__main__":
    main()
