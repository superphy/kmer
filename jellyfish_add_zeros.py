import subprocess
from sklearn import svm
from random import randrange
import os
from math import ceil
from numpy import base_repr
import lmdb
import sys
from timeit import default_timer as timer

#Call the jellyfish command line function
#count: count kmers
#-m 21: counting 21mers
#-s 100M: use a hash with 100 million arguments
#-t 10: use 10 threads
#-C: canonical -- i.e. treat reverse complements as the same: TTT == AAA
#/home/james/Human/F4586: input file
#-o <>: specify output file, default mer_counts.jf
# subprocess.call('jellyfish count -m 3 -s 100M -t 10 /home/james/Bovine/107-1-1.fasta', shell=True)

human_path = '/home/rboothman/Data/human_bovine/human/'
bovine_path = '/home/rboothman/Data/human_bovine/bovine/'

def fasta_to_kmer(filename, k):
    #'Count kmers'
    args = ['jellyfish', 'count', '-m','%d'%k, '-s','10M', '-t','30', '-C', '%s'%filename, '-o','test.jf']
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    #'Get results from kmer count'
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    #'Transform results into usable format'
    arr = [x.split(' ') for x in out.split('\n') if x]

    environment = lmdb.open('kmer_counts', map_size=int(2e9), max_dbs=12)

    database = environment.open_db('k%d'%k, dupsort=False)

    with environment.begin(write=True, db=database) as transaction:
        #'Add the new kmer counts to the database'
        for line in arr:
            transaction.put(line[0], line[1])

        #'Convert Database to list'
        arr = []
        cursor = transaction.cursor()
        for key, value in cursor: #SLOW
            arr.append(int(value))
            cursor.put(key, '0', overwrite=True)

    environment.close()
    print len(arr)
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
    human_files = os.listdir(human_path)
    human_files = [human_path + x for x in human_files]

    bovine_files = os.listdir(bovine_path)
    bovine_files = [bovine_path + x for x in bovine_files]

    print "\nGenerating Training Data....\n"
    data, answers = prepare_data_sub(human_files, bovine_files, k)

    return data, answers

def main():
    start_time = timer()

    k = int(sys.argv[1])
    print "Preparing Data...."
    X,Y = prepare_data(k)

    Z = X[35::-1]
    ZPrime = Y[35::-1]
    X = X[0::34]
    Y = Y[0::34]

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
    end_time = timer()

    print "Time: %d"%(end_time-start_time)

if __name__ == "__main__":
    main()
