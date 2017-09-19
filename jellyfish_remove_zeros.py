import subprocess
from sklearn import svm
from random import randrange
import os
import lmdb
import sys
from timeit import default_timer as timer

human_train_path = '/home/rylan/Data/human_bovine/human_train/'
bovine_train_path = '/home/rylan/Data/human_bovine/bovine_train/'
human_test_path = '/home/rylan/Data/human_bovine/human_test/'
bovine_test_path = '/home/rylan/Data/human_bovine/bovine_test/'

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


def main():
    start_time = timer()

    environment = lmdb.open('database', map_size=int(2e9))

    database = environment.open_db(dupsort=False)

    with environment.begin(write=True, db=database) as transaction:
        k = int(sys.argv[1])

        print "Setting up files...."
        human_train_files = os.listdir(human_train_path)
        human_train_files = [human_train_path + x for x in human_train_files]

        bovine_train_files = os.listdir(bovine_train_path)
        bovine_train_files = [bovine_train_path + x for x in bovine_train_files]

        human_test_files = os.listdir(human_test_path)
        human_test_files = [human_test_path + x for x in human_test_files]

        bovine_test_files = os.listdir(bovine_test_path)
        bovine_test_files = [bovine_test_path + x for x in bovine_test_files]
        print "Done Setting up files"

        counter = 0
        print "Starting first pass...."
        human_train_arrays = []
        human_train_arrays.append(start(human_train_files[0], k, transaction, database))
        human_train_files.pop(0)
        counter += 1
        print "Done %d"%counter
        for filename in human_train_files:
            human_train_arrays.append(firstpass(filename, k, transaction))
            counter += 1
            print "Done %d"%counter
        bovine_train_arrays = []
        for filename in bovine_train_files:
            bovine_train_arrays.append(firstpass(filename, k, transaction))
            counter += 1
            print "Done %d"%counter
        human_test_arrays = []
        for filename in human_test_files:
            human_test_arrays.append(firstpass(filename, k, transaction))
            counter += 1
            print "Done %d"%counter
        bovine_test_arrays = []
        for filename in bovine_test_files:
            bovine_test_arrays.append(firstpass(filename, k, transaction))
            counter += 1
            print "Done %d"%counter
        print "Done first pass"

        print "Starting second pass...."
        counter = 0
        bovine_test_arrays[-1] = second_start(bovine_test_arrays[-1], k, transaction, database)
        counter += 1
        print "Done %d"%counter
        i = len(bovine_test_arrays)-2
        while i >= 0:
            bovine_test_arrays[i] = secondpass(bovine_test_arrays[i], k, transaction)
            i-=1
            counter += 1
            print "Done %d"%counter
        i = len(human_test_arrays)-1
        while i >= 0:
            human_test_arrays[i] = secondpass(human_test_arrays[i], k, transaction)
            i-=1
            counter += 1
            print "Done %d"%counter
        i = len(bovine_train_arrays)-1
        while i >= 0:
            bovine_train_arrays[i] = secondpass(bovine_train_arrays[i], k, transaction)
            i-=1
            counter += 1
            print "Done %d"%counter
        i = len(human_train_arrays)-1
        while i >= 0:
            human_train_arrays[i] = secondpass(human_train_arrays[i], k, transaction)
            i-=1
            counter += 1
            print "Done %d"%counter
        print "Done secondpass"

        print "Shuffling input...."
        training, answers = shuffle(human_train_arrays, bovine_train_arrays, "Human", "Bovine")

        test, test_answers = shuffle(human_test_arrays, bovine_test_arrays, "Human", "Bovine")
        print "Done Shuffling input"

        machine = svm.SVC()
        print "Training SVM...."
        machine.fit(training, answers)
        print "Making Predictions...."
        output = machine.predict(test)

        count = 0
        for i in range(len(test)):
            if test_answers[i] == output[i]:
                count+=1

        ans = (count*100)//len(test)
        print "Percent Correct: %d"%ans

    environment.close()

    end_time = timer()
    print "Time: %d" % (end_time-start_time)

if __name__ == "__main__":
    main()
