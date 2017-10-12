import subprocess
import os
import lmdb
import sys



def __start(filename, k, limit, env, txn, data):
    """
    Performs a kmer count on filename, counting kmers with a length of k and
    removing any kmer that has a count less than limit. Resets the master
    database data and then writes each kmer as a key with value -1 to data.
    Creates a new database called filename, writes each kmer/count pair to the
    new databse as a key value pair.
    """
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

    txn.drop(data, delete=False)
    string = '%s'%filename
    current = env.open_db(string, txn=txn)

    for line in arr:
        txn.put(line[0], line[1])
        txn.put(line[0], line[1], db=current)



def __firstpass(filename, k, limit, env, txn):
    """
    Performs a kmer count on filename, counting kmers with a length of k and
    removing any kmer that has a count less than limit. Creates a new database
    called filename and writes each kmer/count pair from the kmer count to the
    new database. Only writes kmers that are already present in the master
    database that txn points to. Removes any kmer from the master database that
    is not present in filename.
    """
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

    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    txn.drop(current, delete= False)

    for line in arr:
        if txn.get(line[0], default=False):
            txn.put(line[0], line[1], overwrite=True, dupdata=False)

    cursor = txn.cursor()
    for key, value in cursor:
        if value == '-1':
            txn.delete(key)
        else:
            txn.put(key, value, db=current)
            txn.put(key, '-1')



def __secondstart(filename, k, env, txn, data):
    """
    Resets the master database so that it matches the database named filename.
    """
    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    txn.drop(data, delete=False)
    cursor = txn.cursor(db=current)
    for key, val in cursor:
        txn.put(key, val)



def __secondpass(filename, k, env, txn):
    """
    Removes every kmer from the database named filename that is not present in
    the master database.
    """
    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    cursor = txn.cursor(db = current)

    for key, val in cursor:
        if not txn.get(key, default=False):
            txn.delete(key, val, db = current)



def __print_status(counter, total):
    """
    Outputs a progress bar.
    """
    percent = (counter*100)/total
    sys.stdout.write('\r')
    sys.stdout.write("[%-44s] %d%%" % ('='*((percent*44)/100), percent))
    sys.stdout.flush()




def __setup_data(files, k, limit, env, txn, data):
    """
    Takes a list of paths to fasta files, a kmer length, a lower limit on how
    many times a kmer needs to occur in order for it to be output, and an lmdb
    environment, transaction and database.
    """
    counter = 0
    total = len(files)
    print "First Pass"
    __start(files[0], k, limit, env, txn, data)
    temp = files.pop(0)
    counter += 1
    for filename in files:
        __print_status(counter, total)
        __firstpass(filename, k, limit, env, txn)
        counter += 1

    files.insert(0, temp)

    __print_status(counter, total)
    print "\nSecond Pass"
    counter = 0
    __secondstart(files[-1], k, env, txn, data)
    counter += 1
    i = len(files)-2
    while i >= 0:
        __print_status(counter, total)
        __secondpass(files[i], k, env, txn)
        i-=1
        counter += 1

    __print_status(counter, total)
    print "\n"



def count_kmers(k, limit, files, database):
    """
    Counts all kmers of length "k" in the fasta files "files", removing any that
    appear fewer than "limit" times. Stores the output in a lmdb database named
    "database".
    """
    env = lmdb.open('%s'%database, map_size=int(160e9), max_dbs=400)
    data = env.open_db('master', dupsort=False)

    with env.begin(write=True, db=data) as txn:

        __setup_data(files, k, limit, env, txn, data)

    env.close()



def get_counts(files, database):
    """
    Returns (as an array) the kmer counts of each fasta file in "files"
    contained in the lmdb database named "database". The length and lower limit
    of the kmer counts will be the same as when they were calculated using
    count_kmers.
    """
    env = lmdb.open('%s'%database, map_size=int(160e9), max_dbs=400)
    data = env.open_db('master', dupsort=False)

    with env.begin(write=False, db=data) as txn:

        arrays = []

        for f in files:
            array = []
            current = env.open_db('%s'%f, txn = txn)
            cursor = txn.cursor(db=current)
            for key, val in cursor:
                array.append(int(val))

            arrays.append(array)

    env.close()
    return arrays
