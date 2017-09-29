import subprocess
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit as ssSplit
import os
import lmdb
import sys

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

def print_status(counter, total):
    percent = (counter*100)/total
    sys.stdout.write('\r')
    sys.stdout.write("[%-44s] %d%%" % ('='*((percent*44)/100), percent))
    sys.stdout.flush()

def run(k, limit, num_splits):

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

        arrays = []
        arrays.append(start(files[0], k, limit, txn, data))
        files.pop(0)
        counter += 1

        for filename in files:
            print_status(counter, total)
            arrays.append(firstpass(filename, k, limit, txn))
            counter += 1

        print_status(counter, total)
        print ""

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

        sss = ssSplit(n_splits=num_splits, test_size=0.2, random_state=42)

        linear = svm.SVC(kernel='linear')
        rbf = svm.SVC(kernel='rbf' ,gamma=float(1e-09), C=10000.0)
        poly4 = svm.SVC(kernel='poly', degree=4, coef0=0.0)
        default = svm.SVC()


        linearA = []
        linearB = []
        rbfC = []
        poly4C = []
        defaultC = []

        i = 1

        for indices in sss.split(arrays, labels):
            X = [arrays[x] for x in indices[0]]
            Y = [labels[x] for x in indices[0]]
            Z = [arrays[x] for x in indices[1]]
            ZPrime = [labels[x] for x in indices[1]]

            # Scale all data to range [0,1] since SVMs are not scale invariant
            scalerA = MinMaxScaler()
            scalerB = StandardScaler()
            Xa = scalerA.fit_transform(X)
            Za = scalerA.transform(Z)

            Xb = scalerB.fit_transform(X)
            Zb = scalerB.transform(Z)

            try:
                linear.fit(Xa, Y)
                ans = linear.score(Za, ZPrime)
                linearA.append(ans)
                print "linear MinMax: %f" % ans

                linear.fit(Xb, Y)
                ans = linear.score(Zb, ZPrime)
                linearB.append(ans)
                print "linear standard: %f" % ans

                rbf.fit(X, Y)
                ans=rbf.score(Z, ZPrime)
                rbfC.append(ans)
                print "RBF no scaling: %f" % ans

                poly4.fit(X, Y)
                ans=poly4.score(Z, ZPrime)
                poly4C.append(ans)
                print "poly degree 4 no scaling: %f" % ans

                default.fit(X, Y)
                ans=default.score(Z, ZPrime)
                defaultC.append(ans)
                print "default no scaling: %f" % ans

            except (ValueError, TypeError) as E:
                print E
                return -1 -1

    env.close()

    return [sum(linearA)/num_splits, sum(linearB)/num_splits, sum(rbfC)/num_splits,
            sum(poly4C)/num_splits, sum(defaultC)/num_splits]

def get_method(index):
    if index == 0:
        return 'linear MinMaxScaler'
    elif index == 1:
        return 'linear StandardScaler'
    elif index == 2:
        return 'RBF Not scaled'
    elif index == 3:
        return 'Default Not Scaled'
    elif index == 4:
        return 'Poly degree 4 Not Scaled'

def main():
    if len(sys.argv) == 4:
        k = int(sys.argv[1])
        l = int(sys.argv[2])
        rep = int(sys.argv[3])
        output = run(k, l, rep)
        labels = [get_method(i) for i in range(len(output))]
        string = ''
        for i in range(len(output)):
            string += labels[i] + ': ' + str(output[i]) + '\n'
        print string

    else:
        output = """
        Error: Wrong number of arguments, requires exactly 3
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
