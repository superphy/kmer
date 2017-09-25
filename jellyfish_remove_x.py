import subprocess
from sklearn import svm
from random import randrange
import os
import lmdb
import sys
from timeit import default_timer as timer

human_path = '/home/rboothman/Data/human_bovine/human/'
bovine_path = '/home/rboothman/Data/human_bovine/bovine/'

def start(filename, k, limit, txn, data):
    args = ['jellyfish','count','-m','%d'%k,'-s','10M','-t','30','-C','%s'%filename,'-o','test.jf','-L','%d'%limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    txn.drop(data)
    for line in arr:
        txn.put(line[0], line[1])
    array = []
    cursor = txn.cursor()
    #array = [[key, val] for key, val in txn.cursor()]
    for key, value in cursor:
        array.append([key, value])
        txn.put(key, '-1')

    return array

def firstpass(filename, k, limit, txn):
    args = ['jellyfish','count','-m','%d'%k,'-s','10M','-t','30','-C','%s'%filename,'-o','test.jf','-L','%d'%limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    for line in arr:
        if txn.get(line[0], default=False):
            txn.put(line[0], line[1], overwrite=True, dupdata=False)

    array = []
    cursor = txn.cursor()
    for key, value in cursor:
        if value == '-1':
            txn.delete(key)
        else:
            array.append([key, value])
            txn.put(key, '-1')

    return array

def second_start(arr, k, txn, data):
    txn.drop(data)
    for line in arr:
        txn.put(line[0], line[1])
    array = []
    cursor = txn.cursor()
    for key, value in cursor:
        array.append(int(value))

    return array

def secondpass(arr, k, txn):
    for line in arr:
        if txn.get(str(line[0]), default=False):
            txn.put(line[0], line[1], overwrite=True, dupdata=False)

    array = []
    cursor = txn.cursor()
    for key, value in cursor:
        array.append(int(value))

    return array

def shuffle(arrA, arrB, A, B):
    countA = 0
    countB = 0
    test = []
    answers = []

    while countA < len(arrA) and countB < len(arrB):
        rand = randrange(0,10)
        if rand%2 == 0:
            test.append(arrA[countA])
            answers.append(A)
            countA+=1
        else:
            test.append(arrB[countB])
            answers.append(B)
            countB+=1
    while countA < len(arrA):
        test.append(arrA[countA])
        answers.append(A)
        countA+=1
    while countB < len(arrB):
        test.append(arrB[countB])
        answers.append(B)
        countB+=1
    return test, answers

def output(counter):
    sys.stdout.write('\r')
    sys.stdout.write("[%-44s] %d%%" % ('='*(counter/2), (counter*100)/88))
    sys.stdout.flush()

def run(k, limit):
    start_time = timer()

    env = lmdb.open('database', map_size=int(2e9))

    data = env.open_db(dupsort=False)

    with env.begin(write=True, db=data) as txn:

        human_files = os.listdir(human_path)
        human_files = [human_path + x for x in human_files]

        bovine_files = os.listdir(bovine_path)
        bovine_files = [bovine_path + x for x in bovine_files]

        counter = 0

        print "First Pass"
        human_arrays = []
        human_arrays.append(start(human_files[0], k, limit, txn, data))
        human_files.pop(0)
        counter += 1

        for filename in human_files:
            if counter%2 == 0:
                output(counter)
            human_arrays.append(firstpass(filename, k, limit, txn))
            counter += 1

        bovine_arrays = []
        for filename in bovine_files:
            if counter%2 == 0:
                output(counter)
            bovine_arrays.append(firstpass(filename, k, limit, txn))
            counter += 1

        output(counter)

        print "\nSecond Pass"
        counter = 0
        bovine_arrays[-1] = second_start(bovine_arrays[-1], k, txn, data)
        counter += 1

        i = len(bovine_arrays)-2
        while i >= 0:
            if counter%2 == 0:
                output(counter)
            bovine_arrays[i] = secondpass(bovine_arrays[i], k, txn)
            i-=1
            counter += 1

        i = len(human_arrays)-1
        while i >= 0:
            if counter%2 == 0:
                output(counter)
            human_arrays[i] = secondpass(human_arrays[i], k, txn)
            i-=1
            counter += 1

        output(counter)
        print "\n"

        X, Y = shuffle(human_arrays, bovine_arrays, "Human", "Bovine")

        cutoff = int(0.2*len(X))

        Z = X[-cutoff:]
        ZPrime = Y[-cutoff:]
        X = X[:-cutoff]
        Y = Y[:-cutoff]

        machine = svm.SVC()
        try:
            machine.fit(X, Y)
            ans = 100*machine.score(Z, ZPrime)

        except (ValueError, TypeError) as E:
            print E
            return -1 -1

    env.close()

    end_time = timer()

    return ans, (end_time-start_time)

def main():
    k = int(sys.argv[1])
    l = int(sys.argv[2])
    pc, time = run(k, l)
    print "Percent Correct: %d"%pc
    print "Time Elapsed: %d"%time

if __name__ == "__main__":
    main()
