import lmdb
import sys
import os
import subprocess
import numpy as np
import pandas as pd
from threading import Thread
import tempfile

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
    print('Counted kmers for {}'.format(input_file))

def count_files(infiles, outfiles,  k, verbose):
    threads = []
    for i, v in enumerate(infiles):
        threads.append(Thread(target=count_file, args=(v, outfiles[i], k, verbose)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def add_file(input_file, key, env, global_counts, file_counts):
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
    print('Added {} to DB'.format(key))

def backfill_file(db_key, env, global_counts):
    current = env.open_db(db_key.encode())
    with env.begin(write=True, db=current) as txn:
        with txn.cursor(db=global_counts) as cursor:
            for key, value in cursor:
                if not txn.get(key, default=False, db=current):
                    txn.put(key, '0'.encode(), db=current)
    print('Backfilled {}'.format(db_key))

def make_output(db_key, env):
    current = env.open_db(db_key.encode())
    with env.begin(write=True, db=current) as txn:
        num_kmers = txn.stat(current)['entries']
        output = np.zeros(num_kmers, dtype='float64')
        with txn.cursor(db=current) as cursor:
            for index, (key, value) in enumerate(cursor):
                output[index] = float(value)
        txn.put('complete'.encode(), output.tostring(), db=current)
    print('Made output key for {}'.format(db_key))

def make_db(files, db_keys, database, verbose):
    env = lmdb.open(database, map_size=int(160e10), max_dbs=4000)
    global_counts = env.open_db('global_counts'.encode())
    file_counts = env.open_db('file_counts'.encode())

    threads = []
    for i, f in enumerate(files):
        with env.begin(write=False) as txn:
            if not txn.get(db_keys[i].encode(), default=False)
                args = [f, db_keys[i], env, global_counts, file_counts]
                threads.append(Thread(target=add_file, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with env.begin(write=False, db=global_counts) as txn:
        total_kmers = txn.stat()['entries']

    threads = []
    for k in db_keys:
        curr_db = env.open_db(k.encode())
        with env.begin(write=False, db=curr_db) as txn:
            curr_kmers = txn.stat()['entries']
        if curr_kmers < total_kmers:
            args = [k, env, global_counts]
            threads.append(Thread(target=backfill_file, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    threads = []
    for k in db_keys:
        curr_db = env.open_db(k.encode())
        with env.begin(write=False, db=curr_db) as txn:
            curr_kmers = txn.stat()['entries']
        if curr_kmers <= total_kmers:
            threads.append(Thread(target=make_output, args=[k, env]))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    env.close()

def get_output_files(input_files, directory, k):
    db_keys = [x.split('/')[-1].split('.')[0] for x in input_files]
    output_files = [directory + x + '_k{}.csv'.format(k) for x in db_keys]
    database = directory + 'complete_{}-mer_DB'.format(k)
    return output_files, db_keys, database

def count_kmers(k, files, directory, verbose):
    output_files, db_keys, database = get_output_files(files, directory, k)
    needs_counting_in = []
    needs_counting_out = []
    for i, v in enumerate(output_files):
        if not os.path.exists(v):
            needs_counting_out.append(v)
            needs_counting_in.append(files[i])
    count_files(needs_counting_in, needs_counting_out, k, verbose)
    make_db(output_files, db_keys, database, verbose)

def filter_kmers(input_db, output_db, files, k, min_global_count=0,
                 max_global_count=None, min_file_count=0,
                 max_file_count=None):
    max_file_count = max_file_count or len(files)
    max_global_count = max_global_count or (4**k)

    input_env = lmdb.open(input_db, map_size=int(160e10), max_dbs=4000))
    output_env = lmdb.open(output_db, map_size=int(160e10), max_dbs=4000))

    global_counts = input_env.open_db('gloabl_counts'.encode())
    file_counts = input_env.open_db('file_counts'.encode())
    valid_kmers = []
    with input_env.begin(write=False, db=global_counts) as global_txn:
        with global_txn.cursor() as global_cursor:
            for key, global_value in global_cursor:
                if global_value <= max_global_count and global_value >= min_global_count:
                    valid_kmers.append(key)
    with input_env.begin(write=False, db=file_counts) as file_txn:
        for index, key in enumerate(valid_kmers):
            file_value = file_txn.get(key)
                if not (file_value <= max_file_count and file_value >= min_file_count):
                    valid_kmers.pop(index)
    threads = []
    for f in files:
        args = [f, valid_kmers, output_env]
        threads.append(Thread(target=make_output, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def get_counts(k, files, directory):
    output_files, db_keys, database = get_output_files(files, directory, k)
    env = lmdb.open(str(database), map_size=int(160e10), max_dbs=4000)
    global_counts = env.open_db('global_counts'.encode(), create=False)
    with env.begin(write=False, db=global_counts) as txn:
        arrays = []
        for index, value in enumerate(db_keys):
            current = env.open_db(value.encode(), txn=txn)
            complete = txn.get('complete'.encode(), db=current)
            complete = np.fromstring(complete, dtype='float64')
            arrays.append(complete)
        output = np.vstack(arrays)
    env.close()
    return output

def get_kmer_names(k, directory):
    temp, db_keys, database = get_output_files([], directory, k)
    env = lmdb.open(str(database), map_size=int(160e10), max_dbs=4000)

    global_counts = env.open_db('global_counts'.encode(), create=False)
    with env.begin(write=False, db=global_counts) as txn:
        kmer_list = []
        with txn.cursor(db=global_counts) as cursor:
            for item in cursor:
                kmer_list.append(item[0].decode())
    env.close()
    return np.asarray(kmer_list)





