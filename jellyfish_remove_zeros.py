import subprocess
from sklearn import svm
from random import randrange
import os
import lmdb
import sys
from timeit import default_timer as timer

human_path = '/home/rboothman/Data/human_bovine/human/'
bovine_path = '/home/rboothman/Data/human_bovine/bovine/'

def start(filename, k, transaction, database):
    args = ['jellyfish', 'count', '-m','%d'%k, '-s','10M', '-t','30', '-C', '%s'%filename, '-o','test.jf']
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    #'Get results from kmer count'
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    #'Transform results into usable format'
    arr = [x.split(' ') for x in out.split('\n') if x]

    transaction.drop(database)
    for line in arr:
        transaction.put(line[0], line[1])
    array = []
    cursor = transaction.cursor()
    for key, value in cursor:
        array.append([key, value])
        transaction.put(key, '-1')

    return array

def firstpass(filename, k, transaction):
    args = ['jellyfish', 'count', '-m','%d'%k, '-s','10M', '-t','30', '-C', '%s'%filename, '-o','test.jf']
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    #'Get results from kmer count'
    args = ['jellyfish', 'dump', '-c', 'test.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE)
    out, err = p.communicate()
    #'Transform results into usable format'
    arr = [x.split(' ') for x in out.split('\n') if x]

    for line in arr:
        if transaction.get(line[0], default=False):
            transaction.put(line[0], line[1], overwrite=True, dupdata=False)

    array = []
    cursor = transaction.cursor()
    for key, value in cursor:
        if value == '-1':
            transaction.delete(key)
        else:
            array.append([key, value])
            transaction.put(key, '-1')

    return array

def second_start(arr, k, transaction, database):
    transaction.drop(database)
    for line in arr:
        transaction.put(line[0], line[1])
    array = []
    cursor = transaction.cursor()
    for key, value in cursor:
        array.append(int(value))

    return array

def secondpass(arr, k, transaction):
    for line in arr:
        if transaction.get(str(line[0]), default=False):
            transaction.put(line[0], line[1], overwrite=True, dupdata=False)

    array = []
    cursor = transaction.cursor()
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


def run(k):
    start_time = timer()

    environment = lmdb.open('database', map_size=int(2e9))

    database = environment.open_db(dupsort=False)

    with environment.begin(write=True, db=database) as transaction:

        print "Setting up files...."
        human_files = os.listdir(human_path)
        human_files = [human_path + x for x in human_files]

        bovine_files = os.listdir(bovine_path)
        bovine_files = [bovine_path + x for x in bovine_files]

        print "Done Setting up files"

        counter = 0
        print "Starting first pass...."
        human_arrays = []
        human_arrays.append(start(human_files[0], k, transaction, database))
        human_files.pop(0)
        counter += 1
        print "Done %d"%counter
        for filename in human_files:
            human_arrays.append(firstpass(filename, k, transaction))
            counter += 1
            print "Done %d"%counter
        bovine_arrays = []
        for filename in bovine_files:
            bovine_arrays.append(firstpass(filename, k, transaction))
            counter += 1
            print "Done %d"%counter

        print "Done first pass"

        print "Starting second pass...."
        counter = 0
        bovine_arrays[-1] = second_start(bovine_arrays[-1], k, transaction, database)
        counter += 1
        print "Done %d"%counter
        i = len(bovine_arrays)-2
        while i >= 0:
            bovine_arrays[i] = secondpass(bovine_arrays[i], k, transaction)
            i-=1
            counter += 1
            print "Done %d"%counter
        i = len(human_arrays)-1
        while i >= 0:
            human_arrays[i] = secondpass(human_arrays[i], k, transaction)
            i-=1
            counter += 1
            print "Done %d"%counter

        print "Done secondpass"

        print "Shuffling input...."
        X, Y = shuffle(human_arrays, bovine_arrays, "Human", "Bovine")

        print "Done Shuffling input"

        Z = X[35::-1]
        ZPrime = Y[35::-1]
        X = X[0::34]
        Y = Y[0::34]

        machine = svm.SVC()
        print "Training SVM...."
        machine.fit(X, Y)
        print "Making Predictions...."
        output = machine.predict(Z)

        count = 0
        for i in range(len(Z)):
            if ZPrime[i] == output[i]:
                count+=1

        ans = (count*100)//len(Z)

    environment.close()

    end_time = timer()
    return ans, (end_time-start_time)

def main():
    k = int(sys.argv[1])
    pc, time = run(k)
    print "Percent Correct: %d"%pc
    print "Time Elapsed: %d"%time

if __name__ == "__main__":
    main()
