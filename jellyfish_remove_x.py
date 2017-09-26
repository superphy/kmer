import subprocess
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit as ssSplit
from sklearn.model_selection import GridSearchCV
from random import randrange
import os
import lmdb
import sys
from timeit import default_timer as timer

human_path = '/home/rboothman/Data/human_bovine/human/'
bovine_path = '/home/rboothman/Data/human_bovine/bovine/'

def start(filename, k, limit, txn, data):
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30', '-C',
            '%s' % filename, '-o', 'test.jf', '-L', '%d' % limit]
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
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30', '-C',
            '%s' % filename, '-o', 'test.jf', '-L', '%d' % limit]
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

def print_status(counter, total):
    percent = (counter*100)/total
    sys.stdout.write('\r')
    sys.stdout.write("[%-44s] %d%%" % ('='*((percent*44)/100), percent))
    sys.stdout.flush()

def run(k, limit):
    start_time = timer()

    env = lmdb.open('database', map_size=int(2e9))

    data = env.open_db(dupsort=False)

    with env.begin(write=True, db=data) as txn:

        h_files = os.listdir(human_path)
        h_files = [human_path + x for x in h_files]

        b_files = os.listdir(bovine_path)
        b_files = [bovine_path + x for x in b_files]

        files = h_files + b_files

        counter = 0
        total = len(files)

        print "First Pass"
        arrays = []
        arrays.append(start(files[0], k, limit, txn, data))
        files.pop(0)
        counter += 1

        for filename in files:
            print_status(counter, total)
            arrays.append(firstpass(filename, k, limit, txn))
            counter += 1

        print_status(counter, total)

        print "\nSecond Pass"
        counter = 0
        arrays[-1] = second_start(arrays[-1], k, txn, data)
        counter += 1

        i = len(arrays)-2
        while i >= 0:
            print_status(counter, total)
            arrays[i] = secondpass(arrays[i], k, txn)
            i-=1
            counter += 1

        print_status(counter, total)
        print "\n"

        labels=["H" for x in range(len(h_files))]+["B" for x in range(len(b_files))]

        sss = ssSplit(n_splits=5, test_size=0.2, random_state=42)

        for indices in sss.split(arrays, labels):
            X = [arrays[x] for x in indices[0]]
            Y = [labels[x] for x in indices[0]]
            Z = [arrays[x] for x in indices[1]]
            ZPrime = [labels[x] for x in indices[1]]

            # Scale all data to range [0,1] since SVMs are not scale invariant
            scaler = MinMaxScaler()
            print "Scaling data"
            X = scaler.fit_transform(X)
            Z = scaler.transform(Z)

            print "Creating Support Vector Machines"
            linear = svm.SVC(kernel='linear')

            c_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            param_grid = dict(gamma=gamma_range, c=c_range)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(arrays, labels)

            gamma = grid.best_params_['gamma']
            C = grid.best_params_['gamma']

            rbf = svm.SVC(kernel='rbf' ,gamma=gamma, C=c)

            try:
                print "Training Linear Machine..."
                linear.fit(X, Y)
                linear_ans = 100*linear.score(Z, ZPrime)
                print linear_ans

                print "Training RBF Macine..."
                rbf.fit(X, Y)
                rbf_ans = 100*rbf.score(Z, ZPrime)
                print rbf_ans

            except (ValueError, TypeError) as E:
                print E
                return -1 -1

    env.close()

    end_time = timer()

    return linear_ans, rbf_ans

def main():
    if len(sys.argv) == 3:
        k = int(sys.argv[1])
        l = int(sys.argv[2])
        la, rb = run(k, l)
        print "Linear: %d%%" % la
        print "RBF: %d%%" % rb
    else:
        output = """
        Error: Not enough arguments, requires exactly 2
        First Argument: kmer length
        Second Argument: Minimum kmer count required for a kmer to be output

        Example: python %s 10 5
        Counts kmers with length 10 removing any that appear fewer than 5 times
        """ % sys.argv[0]
        print output

if __name__ == "__main__":
    main()
