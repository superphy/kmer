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
from kmerprediction.complete_kmer_counter import KmerCounterError
import logging
import tempfile
from threading import Thread

def count_file(input_file, output_file, k, limit):
    handle, jf_file = tempfile.mkstemp()
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(input_file), '-o', str(jf_file), '-L', '%d' % limit]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    args = ['jellyfish', 'dump', '-c', '-t', str(jf_file), '-o', str(output_file)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    os.remove(jf_file)
    logging.info('Counted kmers for {}'.format(input_file))


def start(input_file, k, limit, env, txn, master):
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
    handle, temp_file = tempfile.mkstemp()
    count_file(input_file, temp_file, k, limit)

    current = env.open_db(input_file.encode(), txn=txn)
    txn.drop(master, delete=False)

    with open(temp_file, 'r') as f:
        for line in f:
            kmer = line.split()[0].encode()
            count = str(line.split()[1]).encode()
            txn.put(kmer, count, db=current)
            txn.put(kmer, '-1'.encode(), db=master)
    os.remove(temp_file)


def firstpass(filename, k, limit, env, master):
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
    handle, temp_file = tempfile.mkstemp()
    count_file(filename, temp_file, k, limit)

    current = env.open_db(filename.encode())

    with env.begin(write=True, db=master) as txn:
        txn.drop(current, delete=False)
        with open(temp_file, 'r') as f:
            for line in f:
                kmer = line.split()[0].encode()
                count = str(line.split()[1]).encode()
                if txn.get(kmer, default=False, db=master):
                    txn.put(kmer, count, db=master)
                    txn.put(kmer, count, db=current)

        with txn.cursor(db=master) as cursor:
            for key, value in cursor:
                if value == '-1'.encode():
                    txn.delete(key)
                else:
                    txn.put(key, '-1'.encode())

    os.remove(temp_file)
    logging.info('Counted kmers for {}'.format(filename))


def secondpass(filename, env, master):
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
    current = env.open_db(filename.encode())
    with env.begin(write=True, db=master) as txn:
        with txn.cursor(db=current) as cursor:
            for key, val in cursor:
                if not txn.get(key, default=False, db=master):
                    txn.delete(key, val, db=current)
    logging.info('Remove missing kmers from {}'.format(filename))


def count_kmers(files, database, k=constants.DEFAULT_K,
                limit=constants.DEFAULT_LIMIT, force=False):
    """
    Counts all kmers of length "k" in the fasta files "files", removing any
    that appear fewer than "limit" times. Stores the output in a lmdb database
    named "database".

    Args:
        k (int):            The length of kmer to count.
        limit (int):        Minimum frequency for a kmer to be output.
        files (list(str)):  The fasta files to count kmers from.
        database (str):     Name of database where the counts will be stored.
        force (bool):       If True all files are recounted, if False only
                            that do not already appear in the database are
                            recounted.

    Returns:
        None
    """
    logging.info('Begin kmer_counter.count_kmers')
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
    master = env.open_db('master'.encode(), dupsort=False)

    recounts = []
    with env.begin(write=True, db=master) as txn:
        logging.info('Begin counting kmers')
        if force or not txn.get(files[0].encode(), default=False):
            start(files[0], k, limit, env, txn, master)
            recounts.append(files[0])

    threads = []
    for filename in files[1:]:
        with env.begin(write=False) as txn:
            if force or not txn.get(filename.encode(), default=False):
                args = [filename, k, limit, env, master]
                threads.append(Thread(target=firstpass, args=args))
                recounts.append(filename)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info('Done counting kmers')

    logging.info('Begin removing missing kmers')
    threads = []
    for filename in files[::-1]:
        with env.begin(write=False) as txn:
            if force or filename in recounts:
                args = [filename, env, master]
                threads.append(Thread(target=secondpass, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info('Done removing missing kmers')

    env.close()
    logging.info('Done kmer_counter.count_kmers')


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
    if not os.path.exists(database):
        msg = 'Attempted to get counts from an uncreated database:'
        msg += ' {}'.format(database)
        raise(KmerCounterError(msg))

    env = lmdb.open(database, map_size=int(160e9), max_dbs=4000, max_readers=1e7)
    try:
        master = env.open_db('master'.encode(), dupsort=False, create=False)
    except lmdb.NotFoundError:
        msg = 'Attempted to get counts from a database that does not contain master'
        msg += ' {}'.format(database)
        logging.exception(msg)
        raise(KmerCounterError(msg))

    if not files:
        output = np.array([], dtype='float64')
    else:
        with env.begin(write=False, db=master) as txn:
            num_keys = txn.stat(master)['entries']
            output = np.zeros((len(files), num_keys), dtype='float64')

            for index, value in enumerate(files):
                try:
                    current = env.open_db(value.encode(), txn=txn, create=False)
                except lmdb.NotFoundError:
                    msg = 'Attempted to get counts for a potentially uncounted'
                    msg += ' genome: {} in DB: {}'.format(value, database)
                    logging.exception(msg)
                    raise(KmerCounterError(msg))

                cursor = txn.cursor(db=current)
                for i, (key, value) in enumerate(cursor):
                    try:
                        output[index, i] = float(value)
                    except IndexError:
                        print(key, value)
                        raise(IndexError)

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

def add_counts(files, database):
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

        for f in files:
            add(f, k, env, txn)

    env.close()
