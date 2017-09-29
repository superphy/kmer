from kmer_prediction import run
import os

human_path = '/home/rboothman/Data/human_bovine/human/'
bovine_path = '/home/rboothman/Data/human_bovine/bovine/'

k_vals = [x for x in range(20, 21)]
lower_limits = [x for x in range(9, 10)]

def setup_files():
    h_files = os.listdir(human_path)
    h_files = [human_path + x for x in h_files]

    b_files = os.listdir(bovine_path)
    b_files = [bovine_path + x for x in b_files]

    return h_files, b_files

with open('TestParamsResults.txt', 'a') as f:
    for k in k_vals:
        for l in lower_limits:
            string = "\nk:\t%d\nl:\t%d\n" % (k, l)
            print string
            f.write(string)
            h_files, b_files = setup_files()
            output = run(k, l, 10, h_files, b_files, None)
            print output
            f.write(str(output))
