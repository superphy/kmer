import lmdb
import sys
import os
import subprocess
import numpy as np
import pandas as pd
from threading import Thread
import tempfile
import shutil
from kmerprediction import constants

class KmerCounterError(Exception):
    """Raise for errors in kmer_counter module"""


def count_file(input_file, output_file, k, verbose):
    handle, temp_file = tempfile.mkstemp()
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(input_file), '-o', str(temp_file)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    args = ['jellyfish', 'dump', '-c', '-t', str(temp_file), '-o', str(output_file)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    os.remove(temp_file)
    if verbose:
        print('Counted kmers for {}'.format(input_file))


def add_file(input_file, key, env, global_counts, file_counts, verbose):
    current = env.open_db(key.encode())
    with env.begin(write=True, db=current) as txn:
        with open(input_file, 'r') as f:
            for line in f:
                kmer = line.split()[0].encode()
                count = str(line.split()[1]).encode()
                txn.put(kmer, count, db=current)

                curr_global_count = txn.get(kmer, default=0, db=global_counts)
                new_global_count = str(int(curr_global_count) + int (count)).encode()
                txn.put(kmer, new_global_count, db=global_counts)

                curr_file_count = txn.get(kmer, default=0, db=file_counts)
                new_file_count = str(1 + int(curr_file_count)).encode()
                txn.put(kmer, new_file_count, db=file_counts)
    if verbose:
        print('Added {} to DB'.format(key))


def backfill_file(db_key, env, global_counts, verbose):
    current = env.open_db(db_key.encode())
    with env.begin(write=True, db=current) as txn:
        with txn.cursor(db=global_counts) as cursor:
            for key, value in cursor:
                if not txn.get(key, default=False, db=current):
                    txn.put(key, '0'.encode(), db=current)
    if verbose:
        print('Backfilled {}'.format(db_key))


def make_output(key, valid_kmers, env, name, verbose):
    db = env.open_db(key.encode())
    with env.begin(write=True, db=db) as txn:
        output = np.zeros(len(valid_kmers), dtype=int)
        for index, kmer in enumerate(valid_kmers):
            output[index] = int(txn.get(kmer, default=0, db=db))
        txn.put(name.encode(), output.tostring(), db=db)
    if verbose:
        print('Made output key for {}'.format(db_key))


def make_db_keys(input_files):
    output = [x.split('/')[-1] for x in input_files]
    output = [x.split('.')[:-1] for x in output]
    output = ['.'.join(x) for x in output]
    return output


def count_kmers(fasta_files, database, k=constants.DEFAULT_K, verbose=True,
                output_db=None, min_global_count=0, max_global_count=None,
                min_file_count=0, max_file_count=None,
                name=constants.DEFAULT_NAME):
    """
    Args:
        k (int):                The length of k-mer to count.
        fasta_files (list):     The paths to each fasta file to count kmers from
        database (str):         Path to the database where the complete results
                                from the kmer count will be stored.
        verbose (bool):         If Ture status messages will be output.
        output_db (str):        Path to a second database to store the a smaller
                                version of the output in, usefull for counting
                                large k-mers and storing the complete db on a
                                slower to access drive.
        min_global_count (int): The minimum number of a times a kemr must appear
                                in the entire database in order to be output.
        max_global_count (int): The maximum number of times a kmer can appear
                                in the entire database in order to be output.
        min_file_count (int):   The minimum number a fasta files that a kmer
                                must appear in inorder to be output.
        max_file_count (int):   The maximum number of fasta files that a kmer
                                can appear in inorder to be outoput.
        name (str):             The name for the key whose value will contain
                                the complete output for a fasta file. User can
                                specify so that multiple filter results can be
                                stored in one DB.
    Returns:
        None
    """
    max_file_count = max_file_count or len(fasta_files)
    max_global_count = max_global_count or (4**k)

    db_keys = make_db_keys(fasta_files)
    temp_dir = tempfile.mkdtemp()
    temp_files = [temp_dir + '/' + x for x in db_keys]

    env = lmdb.open(database, map_size=int(160e10), max_dbs=4000)
    global_counts = env.open_db('global_counts'.encode())
    file_counts = env.open_db('file_counts'.encode())

    # Count kmers of length k for each file in fasta_files
    # Store results in coresponding file in temp_files
    threads = []
    for i, v in enumerate(fasta_files):
        with env.begin(write=False) as txn:
            if not txn.get(db_keys[i].encode(), default=False):
                args = [v, temp_files[i], k, verbose]
                threads.append(Thread(target=count_file, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Add kmer counts from each csv file to DB
    # Don't add kmer counts if the db_key corresponding to the csv_file 
    # already exists as a named database in the DB
    threads = []
    for i, f in enumerate(temp_files):
        with env.begin(write=False) as txn:
            if not txn.get(db_keys[i].encode(), default=False):
                args = [f, db_keys[i], env, global_counts, file_counts, verbose]
                threads.append(Thread(target=add_file, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with env.begin(write=False, db=global_counts) as txn:
        total_kmers = txn.stat()['entries']

    # Backfill missing kmers in DB
    # For each db_key add a count of 0 for each kmer that exists in the DB
    # but not in the current genome.
    # Don't backfill a file if it already has equal number of keys as total
    # kmers in the DB
    threads = []
    for k in db_keys:
        curr_db = env.open_db(k.encode())
        with env.begin(write=False, db=curr_db) as txn:
            curr_kmers = txn.stat()['entries']
        if curr_kmers < total_kmers:
            args = [k, env, global_counts, verbose]
            threads.append(Thread(target=backfill_file, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if output_db:
        output_env = lmdb.open(output_db, map_size=160e10, max_dbs=4000)
    else:
        output_env = env

    # Ensure that only kmers whose file/global counts are within the given
    # min/max range are added to the output db
    global_counts = env.open_db('gloabl_counts'.encode())
    file_counts = env.open_db('file_counts'.encode())
    valid_kmers = []
    with env.begin(write=False, db=global_counts) as global_txn:
        with global_txn.cursor() as global_cursor:
            for key, global_value in global_cursor:
                if global_value <= max_global_count and global_value >= min_global_count:
                    valid_kmers.append(key)
    with env.begin(write=False, db=file_counts) as file_txn:
        for index, key in enumerate(valid_kmers):
            file_value = file_txn.get(key)
            if not (file_value <= max_file_count and file_value >= min_file_count):
                valid_kmers.pop(index)

    valid_kmer_array = np.asarray(valid_kmers, dtype=str)
    with output_env.begin(write=True) as txn:
        txn.put(name.encode(), valid_kmer_array.tostring())

    # Add all valid kmers to the output DB
    threads = []
    for k in db_keys:
        with output_env.begin(write=False) as txn:
            if not txn.get(k.encode(), default=False):
                args = [k, valid_kmers, output_env, name, verbose]
                threads.append(Thread(target=make_output, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    shutil.rmtree(temp_dir)

    env.close()


def get_counts(files, database, name=constants.DEFAULT_NAME):
    db_keys = make_db_keys(files)

    if not os.path.exists(database):
        msg = 'Attempted to get counts from an uncreated database: {}'.format(database)
        raise(KmerCounterError(msg))

    env = lmdb.open(database, map_size=int(160e10), max_dbs=4000)
    with env.begin(write=False) as txn:
        arrays = []
        for index, value in enumerate(db_keys):

            try:
                current = env.open_db(value.encode(), txn=txn, create=False)
            except Exception as e:
                print(e)
                msg = 'Attempted to get counts for potentially uncounted genome:'
                msg += ' {} in DB: {}'.format(value, database)
                raise(KmerCounterError(msg))

            results = txn.get(name.encode(), db=current)
            if results is None:
                msg = 'Attempted to get counts for potentially invalid filter method:'
                msg += ' {} for genome: {} in DB: {}'.format(name, value, database)
                raise(KmerCounterError(msg))

            results = np.fromstring(results, dtype='float64')
            arrays.append(results)
        output = np.vstack(arrays)
    env.close()
    return output


def get_kmer_names(database, name=constants.DEFAULT_NAME):
    env = lmdb.open(database, map_size=int(160e10), max_dbs=4000)
    with env.begin(write=False) as txn:
        output = txn.get(name.encode())
    env.close()
    return np.fromstring(output, dtype=str)


