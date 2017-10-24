import os
import sys
from kmer_prediction import run
from kmer_counter import count_kmers, get_counts

file_path = '/home/rboothman/Data/salmonella_amr/'

key = '/home/rboothman/Data/amr_sorted.csv'

def setup_files(antibiotic, arrays):
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

def main(k_range, l_range, reps):
    all_files = [file_path + x  for x in os.listdir(file_path)]
    k_range = [int(x) for x in k_range.split(',')]
    l_range = [int(x) for x in l_range.split(',')]
    antibiotics = ['ampicillin', 'chloramphenicol', 'gentamicin',
                    'kanamycin', 'nalidixic acid', 'spectinomycin',
                    'streptomycin', 'sulphonamides', 'tetracycline',
                    'trimethoprim']
    for l in l_range:
        for k in k_range:
            count_kmers(k, l, all_files, "database")
            for antibiotic in antibiotics:
                sus, res = setup_files(antibiotic, all_files)
                sus = get_counts(sus, "database")
                res = get_counts(res, "database")
                with open('/home/rboothman/Data/lower_limits/results_salmonella/%s/l%d.txt'%(antibiotic,l), 'a') as f:
                    output = run(k, l, reps, sus, res, None)
                    f.write("%d,%f\n" % (k, output))
                    print antibiotic, k, l, output

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
