import subprocess
import os
import lmdb
import sys
import numpy as np


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
    txn.drop(current, delete=False)

    for line in arr:
        if txn.get(line[0], default=False):
            txn.put(line[0], line[1], overwrite=True, dupdata=False)

    with txn.cursor() as cursor:
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
    with txn.cursor(db=current) as cursor:
        for key, val in cursor:
            txn.put(key, val)



def __secondpass(filename, k, env, txn):
    """
    Removes every kmer from the database named filename that is not present in
    the master database.
    """
    string = '%s'%filename
    current = env.open_db(string, txn=txn)
    with txn.cursor(db=current) as cursor:
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


def __add(filename, k, env, txn):
    """
    Counts kmers in filename and adds them to the database pointed to by env if
    the kmer already exists in env.
    """
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30', '-C',
            '%s' % filename, '-o', 'test.jf']
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
    txn.drop(current, delete=False)

    for line in arr:
        if txn.get(line[0], default=False):
            txn.put(line[0], line[1], overwrite=True, dupdata=False, db=current)

    with txn.cursor() as cursor:
        for key, val in cursor:
            if not txn.get(key, default=False, db=current):
                txn.put(key, '0', overwrite=True, db=current)



def __add_to_database(files, k, env, txn):
    """
    Adds the kmer counts in files to an already created database, does not
    remove any kmer counts from the data base, sets
    """
    counter = 0
    total = len(files)
    print "Begin"
    for file in files:
        __print_status(counter, total)
        __add(file, k, env, txn)
        counter += 1
    __print_status(counter, total)
    print "\n"



def count_kmers(k, limit, files, database):
    """
    Counts all kmers of length "k" in the fasta files "files", removing any that
    appear fewer than "limit" times. Stores the output in a lmdb database named
    "database".
    """
    env = lmdb.open('%s'%database, map_size=int(160e9), max_dbs=4000)
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
    env = lmdb.open('%s'%database, map_size=int(160e9), max_dbs=4000)
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



def add_counts(files, database):
    """
    Counts kmers in the fasta files "files" removing any that do not already
    appear in "database". If a kmer in "files" has a count less than "limit",
    but the kmer appears in database the count will appear in database.
    This function is useful for counting kmers in a new data set that
    you want to make predictions on using an already trained machine learning
    model. The kmer size and cutoff used here will match the kmer size and
    cutoff used when the database was originally created.
    """
    env = lmdb.open('%s'%database, map_size=int(160e9), max_dbs=100000)
    master = env.open_db('master', dupsort=False)

    with env.begin(write=True, db=master) as txn:
        with txn.cursor() as cursor:
            cursor.first()
            key, val = cursor.item()
            k = len(key)

        __add_to_database(files, k, env, txn)

    env.close()
