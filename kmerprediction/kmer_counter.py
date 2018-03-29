"""
Wraps Jellyfish kmer counter to store the results in a database and ensure that
the results can be passed to a machine learning model i.e. if multiple genomes
are passed to jellyfish only the kmers that appear in all the genoms will be
returned this ensures that the data is "rectangular".
"""
from __future__ import division
from __future__ import print_function

from builtins import str
from past.utils import old_div
import subprocess
import sys
import os
import lmdb
import numpy as np
from kmerprediction import constants

def start(filename, k, limit, env, txn, data):
    """
    Performs a kmer count on filename, counting kmers with a length of k and
    removing any kmer that has a count less than limit. Resets the master
    database data and then writes each kmer as a key with value -1 to data.
    Creates a new database called filename, writes each kmer/count pair to the
    new database as a key value pair.

    Args:
        filename (str):             Fasta file to perform a kmer count on
        k (int):                    The length of kmer to count
        limit (int):                Minimum frequency of kmer count to be
                                    output
        env (lmdb.Environment):
        txn (lmdb.Transaction):
        data (Environment handle):

    Returns:
        None
    """
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(filename), '-o', 'counts.jf', '-L', '%d' % limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()
    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'counts.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()
    os.remove('counts.jf')
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    txn.drop(data, delete=False)
    current = env.open_db(filename.encode(), txn=txn)

    for line in arr:
        txn.put(line[0].encode(), line[1].encode())
        txn.put(line[0].encode(), line[1].encode(), db=current)


def firstpass(filename, k, limit, env, txn):
    """
    Performs a kmer count on filename, counting kmers with a length of k and
    removing any kmer that has a count less than limit. Creates a new database
    called filename and writes each kmer/count pair from the kmer count to the
    new database. Only writes kmers that are already present in the master
    database that txn points to. Removes any kmer from the master database that
    is not present in filename.

    Args:
        filename (str):         Fasta file to perform a kmer count on.
        k (int):                Length of kmer to count.
        limit (int):            Minimum frequency of kmer to be output.
        env (lmdb.Environment):
        txn (lmdb.Transaction):

    Returns:
        None
    """
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(filename), '-o', 'counts.jf', '-L', '%d' % limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'counts.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()
    os.remove('counts.jf')
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    current = env.open_db(filename.encode(), txn=txn)
    txn.drop(current, delete=False)

    for line in arr:
        if txn.get(line[0].encode(), default=False):
            txn.put(line[0].encode(), line[1].encode(), overwrite=True, dupdata=False)

    with txn.cursor() as cursor:
        for key, value in cursor:
            if value == '-1'.encode():
                txn.delete(key)
            else:
                txn.put(key, value, db=current)
                txn.put(key, '-1'.encode())


def secondstart(filename, env, txn, data):
    """
    Resets the master database so that it matches the database named filename.

    Args:
        filename (str):             Fasta file to perform a kmer count on.
        k (int):                    Length of kmer to count.
        env (lmdb.Environment):
        txn (lmdb.Transaction):
        data (Environment handle):

    Returns:
        None
    """
    current = env.open_db(filename.encode(), txn=txn)
    txn.drop(data, delete=False)
    with txn.cursor(db=current) as cursor:
        for key, val in cursor:
            txn.put(key, val)


def secondpass(filename, env, txn):
    """
    Removes every kmer from the database named filename that is not present in
    the master database.

    Args:
        filename (str):         Fasta file to perform a kmer count on.
        k (int):                Length of kmer to count.
        env (lmdb.Environment):
        txn (lmdb.Transaction):

    Returns:
        None
    """
    current = env.open_db(filename.encode(), txn=txn)
    with txn.cursor(db=current) as cursor:
        for key, val in cursor:
            if not txn.get(key, default=False):
                txn.delete(key, val, db=current)


def print_status(counter, total, verbose):
    """
    Outputs a progress bar.

    Args:
        counter (int):  Numer of steps completed.
        total (int):    Total number of steps.
        verbose (bool): If False the status bar is not displayed.

    Returns:
        None
    """
    if verbose:
        p = old_div((counter * 100), total)
        sys.stdout.write('\r')
        sys.stdout.write("[%-44s] %d%%" % ('=' * (old_div((p * 44), 100)), p))
        sys.stdout.flush()
        if p == 100:
            print("\n")
    else:
        pass


def setup_data(files, k, limit, env, txn, data, verbose):
    """
    Takes a list of paths to fasta files, a kmer length, a lower limit on how
    many times a kmer needs to occur in order for it to be output, and an lmdb
    environment, transaction and database.

    Args:
        files (list(str)):          The fasta files to count kmers from.
        k (int):                    Length of kmer to count.
        limit (int):                Minimum frequency for a kmer to be output.
        env (lmdb.Environment):
        txn (lmdb.Transaction):
        data (environment handle):
        verbose (bool):             If True display a status bar.

    Returns:
        None
    """
    counter = 0
    total = len(files)

    start(files[0], k, limit, env, txn, data)
    temp = files.pop(0)
    counter += 1
    for filename in files:
        print_status(counter, total, verbose)
        firstpass(filename, k, limit, env, txn)
        counter += 1

    files.insert(0, temp)

    print_status(counter, total, verbose)
    counter = 0
    secondstart(files[-1], env, txn, data)
    counter += 1
    i = len(files) - 2
    while i >= 0:
        print_status(counter, total, verbose)
        secondpass(files[i], env, txn)
        i -= 1
        counter += 1

    print_status(counter, total, verbose)


def add(filename, k, env, txn):
    """
    Counts kmers in filename and adds them to the database pointed to by env.
    Only counts kmers that already exists in env.

    Args:
        filename (str):         Fasta file to perform a kmer count on.
        k (int):                Length of kmer to count.
        env (lmdb.Environment):
        txn (lmdb.Transaction):

    Returns:
        None
    """
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(filename), '-o', 'counts.jf']
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    # Get results from kmer count
    args = ['jellyfish', 'dump', '-c', 'counts.jf']
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()
    os.remove('counts.jf')
    # Transform results into usable format
    arr = [x.split(' ') for x in out.split('\n') if x]

    current = env.open_db(filename.encode(), txn=txn)
    txn.drop(current, delete=False)

    for line in arr:
        if txn.get(line[0].encode(), default=False):
            txn.put(line[0].encode(), line[1].encode(), overwrite=True,
                    dupdata=False, db=current)

    with txn.cursor() as cursor:
        for item in cursor:
            if not txn.get(item[0], default=False, db=current):
                txn.put(item[0], '0'.encode(), overwrite=True, db=current)


def add_to_database(files, k, env, txn, verbose):
    """
    Adds the kmer counts in files to an already created database, does not
    remove any kmer counts from the data base.

    Args:
        files (list(str)):      The fasta files to count kmers from.
        k (int):                The length of kmer to count.
        env (lmdb.Environment):
        txn (lmdb.Transaction):
        verbose (bool):         If True display a status bar.

    Returns:
        None
    """
    counter = 0
    total = len(files)
    for f in files:
        print_status(counter, total, verbose)
        add(f, k, env, txn)
        counter += 1
    print_status(counter, total, verbose)


def count_kmers(files, database, k=constants.DEFAULT_K,
                limit=constants.DEFAULT_LIMIT, verbose=True):
    """
    Counts all kmers of length "k" in the fasta files "files", removing any
    that appear fewer than "limit" times. Stores the output in a lmdb database
    named "database".

    Args:
        k (int):            The length of kmer to count.
        limit (int):        Minimum frequency for a kmer to be output.
        files (list(str)):  The fasta files to count kmers from.
        database (str):     Name of database where the counts will be stored.
        verbose (bool):     If True display a status bar.

    Returns:
        None
    """
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
    data = env.open_db('master'.encode(), dupsort=False)

    with env.begin(write=True, db=data) as txn:

        setup_data(files, k, limit, env, txn, data, verbose)

    env.close()


def get_counts(files, database, name=None):
    """
    Returns (as an array) the kmer counts of each fasta file in "files"
    contained in the lmdb database named "database". The length and lower limit
    of the kmer counts will be the same as when they were calculated using
    count_kmers.

    Args:
        files (list(str)): The fasta files whose kmer counts you would like to
                           retrieve. The kmer counts must already have been
                           calculated using count_kmers.
        database (str):    The databse where the kmer counts are stored.
        name:              Not used, here for compatability.

    Returns:
        list(list): The kmer counts for each genome in files.
    """
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
    data = env.open_db('master'.encode(), dupsort=False)

    with env.begin(write=False, db=data) as txn:
        if files:
            first = env.open_db(files[0].encode(), txn=txn)
        else:
            return np.array([], dtype='float64')
        num_keys = txn.stat(first)['entries']
        output = np.zeros((len(files), num_keys), dtype='float64')

        for index, value in enumerate(files):
            current = env.open_db(value.encode(), txn=txn)
            cursor = txn.cursor(db=current)
            counter = 0
            for item in cursor:
                output[index, counter] = float(item[1])
                counter += 1

    env.close()
    return output


def get_kmer_names(database, name=None):
    """
    Returns (as a numpy 1D array) every key in the databse, this should be an
    alphabetical list of all the kmers in the database.

    Args:
        database (str): The name of the database to get the keys from.
        name:           Not used, here for compatability.

    Returns:
        list(str): Every kmer in the database sorted alphabetically.
    """
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
    data = env.open_db('master'.encode(), dupsort=False)

    with env.begin(write=False, db=data) as txn:

        kmer_list = []
        cursor = txn.cursor()

        for item in cursor:
            kmer_list.append(item[0].decode())

    env.close()
    return np.asarray(kmer_list)


def add_counts(files, database, verbose):
    """
    Counts kmers in the fasta files "files" removing any that do not already
    appear in "database". If a kmer in "files" has a count less than "limit",
    but the kmer appears in database the count will appear in database.
    This function is useful for counting kmers in a new data set that
    you want to make predictions on using an already trained machine learning
    model. The kmer size and cutoff used here will match the kmer size and
    cutoff used when the database was originally created.

    Args:
        files (list(str)): The fasta files containing the genomes whose kmer
                           counts you want added to the database.
        database (str):    The name of the database to add the kmer counts to.
        verbose (bool):    If True display a status bar.

    Returns:
        None
    """
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=100000)
    master = env.open_db('master'.encode(), dupsort=False)

    with env.begin(write=True, db=master) as txn:
        with txn.cursor() as cursor:
            cursor.first()
            item = cursor.item()
            k = len(item[0].decode())

        add_to_database(files, k, env, txn, verbose)

    env.close()
