import subprocess
from sklearn import svm
from random import randrange
import os
from math import ceil
from numpy import base_repr

#Call the jellyfish command line function
#count: count kmers
#-m 21: counting 21mers
#-s 100M: use a hash with 100 million arguments
#-t 10: use 10 threads
#-C: canonical -- i.e. treat reverse complements as the same: TTT == AAA
#/home/james/Human/F4586: input file
#-o <>: specify output file, default mer_counts.jf
# subprocess.call('jellyfish count -m 3 -s 100M -t 10 /home/james/Bovine/107-1-1.fasta', shell=True)

human_train_path = '/home/rylan/Data/human_bovine/human_train/'
bovine_train_path = '/home/rylan/Data/human_bovine/bovine_train/'
human_test_path = '/home/rylan/Data/human_bovine/human_test/'
bovine_test_path = '/home/rylan/Data/human_bovine/bovine_test/'

def fasta_to_kmer(filename, k):
    size = int(ceil((4**k)/(0.75*1000000)))

    args = ['jellyfish', 'count', '-m','%d'%k, '-s','%dM'%size, '-t','20', '-C', '%s'%filename, '-o','test.jf']
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()

    arr = [int(x) for x in ''.join(out).split() if x.isdigit()] #Fast up to a k-value of 10
    return arr

def prepare_data_sub(human, bovine, k):
    h_kmer = []
    b_kmer = []
    for i in range(len(human)):
        print "Counting kmers for human %d/%d" % (i+1, len(human))
        h_kmer.append(fasta_to_kmer(human[i], k))
    for i in range(len(bovine)):
        print "Counting kmers for bovine %d/%d" % (i+1, len(bovine))
        b_kmer.append(fasta_to_kmer(bovine[i], k))
    h_count = 0;
    b_count = 0;
    test = []
    answers = []
    while h_count < len(human) and b_count < len(bovine):
        rand = randrange(0,10)
        if rand%2 == 0:
            test.append(h_kmer[h_count])
            answers.append("Human")
            h_count += 1
        else:
            test.append(b_kmer[b_count])
            answers.append("Bovine")
            b_count += 1
    while h_count < len(human):
        test.append(h_kmer[h_count])
        answers.append("Human")
        h_count += 1
    while b_count < len(bovine):
        test.append(b_kmer[b_count])
        answers.append("Bovine")
        b_count += 1
    return test, answers

def prepare_data(k):
    print "Setting up files...."
    human_train_files = os.listdir(human_train_path)
    human_train_files = [human_train_path + x for x in human_train_files]

    bovine_train_files = os.listdir(bovine_train_path)
    bovine_train_files = [bovine_train_path + x for x in bovine_train_files]

    bovine_test_files = os.listdir(bovine_test_path)
    bovine_test_files = [bovine_test_path + x for x in bovine_test_files]

    human_test_files = os.listdir(human_test_path)
    human_test_files = [human_test_path + x for x in human_test_files]

    print "\nGenerating Training Data....\n"
    training_data, training_answers = prepare_data_sub(human_train_files, bovine_train_files, k)
    print "\nGenerating Test Data....\n"
    test_data, test_answers = prepare_data_sub(human_test_files, bovine_test_files, k)

    return training_data, training_answers, test_data, test_answers


def main():
    print "Preparing Data...."
    X,Y,Z,ZPrime = prepare_data(7)

    print "\nCreating Support Vector Machine....\n"
    machine = svm.SVC()
    print "Training Support Vector Machine...\n"
    machine.fit(X,Y)

    print "Creating Predictions....\n"
    output = machine.predict(Z)

    count = 0
    for i in range(len(Z)):
        if ZPrime[i] == output[i]:
            count += 1

    ans = (count*100)//len(Z)
    print "Percent Correct: %d" % ans

if __name__ == "__main__":
    main()
