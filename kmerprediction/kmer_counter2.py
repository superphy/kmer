import lmdb
import time
import sys
import os
import subprocess
import numpy as np
import threading
from threading import Thread

# counter = 0
# 
# def print_status(counter, total, verbose):
#     if verbose:
#         p = (counter*100)/total
#         sys.stdout.write('\r')
#         sys.stdout.write("[%-44s] %d%%" % ('=' * int((p*44)/100), p))
#         sys.stdout.flush()
#         if p == 100:
#             print("\n")
#     else:
#         pass
# 
# def count_file(filename, k, limit, total, verbose):
#     output_file = filename.split('/')[-1].replace('.fasta', '.jf')
#     args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
#             '-C', str(filename), '-L', '0', '-o', str(output_file)]
#     p = subprocess.Popen(args, bufsize=-1)
#     p.communicate()
#     global counter
#     counter += 1
#     print_status(counter, total, verbose)
# 
# def store_results(filename, env, total, verbose):
#     output_file = filename.split('/')[-1].replace('.fasta', '.jf')
#     args = ['jellyfish', 'dump', '-c', str(output_file)]
#     p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE,
#                          universal_newlines=True)
#     out, err = p.communicate()
# 
#     os.remove(output_file)
# 
#     current = env.open_db(filename.encode(), dupsort=False)
#     master = env.open_db('master'.encode(), dupsort=False)
#     with env.begin(write=True, db=master) as txn:
#         for line in out.split('\n'):
#             if line:
#                 key, value = line.split(' ')
#                 txn.put(key.encode(), value.encode(), overwrite=True, dupdata=False, db=current)
#                 curr_value = txn.get(key.encode(), default=0, db=master)
#                 new_value = str(int(curr_value) + int(value.encode()))
#                 txn.put(key.encode(), new_value.encode(), overwrite=True, dupdata=False, db=master)
#     global counter
#     counter += 1
#     print_status(counter, total, verbose)
# 
# def back_fill(filename, env, total, verbose):
#     data = env.open_db('master'.encode(), dupsort=False)
#     with env.begin(write=True, db=data) as txn:
#         current = env.open_db(filename.encode(), txn=txn)
#         with txn.cursor() as cursor:
#             for key, value in cursor:
#                 if not txn.get(key, default=False, db=current):
#                     txn.put(key, '0'.encode(), db=current)
#     global counter
#     counter += 1
#     print_status(counter, total, verbose)
# 
# def Count_Kmers(k, limit, files, database, verbose):
#     env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
#     total = len(files)
#     threads = []
#     global counter
#     counter = 0
#     t0 = time.time()
#     for f in files:
#         threads.append(Thread(target=count_file, args=[f, k, limit, total, verbose]))
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#     t1 = time.time()
#     counter += 1
#     print_status(counter, total, verbose)
#     print('{}s'.format(t1-t0))
#     counter = 0
#     threads = []
#     t0 = time.time()
#     for f in files:
#         store_results(f, env, total, verbose)
#     t1 = time.time()
#     counter += 1
#     print_status(counter, total, verbose)
#     print('{}s'.format(t1-t0))
#     counter = 0
#     threads = []
#     t0 = time.time()
#     for f in files:
#         threads.append(Thread(target=back_fill, args=[f, env, total, verbose]))
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#     t1 = time.time()
#     counter += 1
#     print_status(counter, total, verbose)
#     print('{}s'.format(t1-t0))
# 
#     env.close()

def firstpass(filename, k, limit, env, txn):
    tempfile = filename + '.jf'
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(filename), '-o', str(tempfile)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    args = ['jellyfish', 'dump', '-c', str(tempfile)]
    p = subprocess.Popen(args, bufsize=-1, stdout=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()

    os.remove(tempfile)

    arr = [x.split(' ') for x in out.split('\n') if x]

    current = env.open_db(filename.encode(), txn=txn)
    txn.drop(current, delete=False)

    for line in arr:
        curr_count = txn.get(line[0].encode(), default=0)
        new_count = str(int(line[1]) + int(curr_count))
        txn.put(line[0].encode(), new_count.encode(), overwrite=True, dupdata=False)
        txn.put(line[0].encode(), line[1].encode(), overwrite=True, dupdata=False, db=current)


def secondpass(filename, env, txn):
    current = env.open_db(filename.encode(), txn=txn)
    with txn.cursor() as cursor:
        for key, value in cursor:
            if not txn.get(key, default=False, db=current):
                txn.put(key, '0'.encode(), db=current)

def print_status(counter, total, verbose):
    if verbose:
        p = (counter*100)/total
        sys.stdout.write('\r')
        sys.stdout.write("[%-44s] %d%%" % ('=' * int((p*44)/100), p))
        sys.stdout.flush()
        if p == 100:
            print("\n")
    else:
        pass


def setup_data(files, k, limit, env, txn, data, verbose):
    counter = 0
    total = len(files)

    for filename in files:
        print_status(counter, total, verbose)
        firstpass(filename, k, limit, env, txn)
        counter += 1

    print_status(counter, total, verbose)
    counter = 0

    for filename in files:
        print_status(counter, total, verbose)
        secondpass(filename, env, txn)
        counter += 1

    print_status(counter, total, verbose)


def count_kmers(k , limit, files, database, verbose):
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)

    info = env.open_db('info'.encode(), dupsort=False)
    with env.begin(write=True, db=info) as txn:
        txn.put('k'.encode(), str(k))
        txn.put('limit'.encode(), str(limit))

    master = env.open_db('master'.encode(), dupsort=False)
    with env.begin(write=True, db=master) as txn:
        setup_data(files, k, limit, env, txn, data, verbose)

    env.close()


def get_counts(k, limit, files, database):
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)

    info = env.open_db('info'.encode(), dupsort=False)
    with env.begin(write=False, db=info) as txn:
        db_k = txn.get('k'.encode(), default=False)
        db_limit = txn.get('limit'.encode(), default=False)

    if (not db_k) or (not db_limit) or (k != db_k) or (limit != db_limit):
        return np.array([], dtype='float64')

    master = env.open_db('master'.encode(), dupsort=False)
    with env.begin(write=False, db=master) as txn:
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


def get_kmer_names(database):
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
    data = env.open_db('master'.encode(), dupsort=False)

    with env.begin(write=False, db=data) as txn:

        kmer_list = []
        cursor = txn.cursor()

        for item in cursor:
            kmer_list.append(item[0].decode())

    env.close()
    return np.asarray(kmer_list)
