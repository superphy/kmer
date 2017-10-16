import os
import sys
from kmer_prediction import run

file_path = '/home/rboothman/Data/salmonella_amr/'

key = '/home/rboothman/Data/amr_sorted.csv'

def setup_files(antibiotic):
    susceptible = []
    resistant = []
    with open(key, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')
        for line in lines:
            if line:
                line = line.split(',')
                if line[1] == antibiotic and not len(line[0]) == 7:
                    if line[2] == "Susceptible":
                        susceptible.append(line[0])
                    else:
                        resistant.append(line[0])

    susceptible = [file_path + x + '.fna' for x in susceptible]
    resistant = [file_path + x + '.fna' for x in resistant]

    return susceptible, resistant

def main(k, l, reps, antibiotic):
    sus, res = setup_files(antibiotic)
    return run(k, l, reps, sus, res, None)

if __name__ == '__main__':
    output = main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    print output
