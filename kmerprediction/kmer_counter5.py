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

def print_status(s1, t1, s2, t2, verbose=True):
    sys.stdout.write('\r')
    sys.stdout.write("{}/{} kmers counted \t {}/{} files done".format(s1, t1, s2, t2))
    sys.stdout.flush()
    if s1 == t1 and s2 == t2:
        print("\n")

def make_db(files, db_keys, database, verbose):
    outer_total = len(files)

    env = lmdb.open(database, map_size=int(160e9), max_dbs=4000)
    global_counts = env.open_db('global_counts'.encode())
    file_counts = env.open_db('file_counts'.encode())

    with env.begin(write=True, db=global_counts) as txn:
        print('Adding counts to DB')
        for i, f in enumerate(files):
            current = env.open_db(db_keys[i].encode(), txn=txn)
            data = pd.read_csv(f, sep='\t', names=['kmer', 'count'])
            inner_total = data.shape[0]
            counter = 0
            print_status(counter, inner_total, i, outer_total, verbose)
            for value in data.itertuples():
                txn.put(value.kmer.encode(), str(value.count).encode(), db=current)

                curr_global_count = txn.get(value.kmer.encode(), default=0, db=global_counts)
                new_global_count = str(value.count + int(curr_global_count))
                txn.put(value.kmer.encode(), new_global_count.encode(), db=global_counts)

                curr_file_count = txn.get(value.kmer.encode(), default=0, db=file_counts)
                new_file_count = str(1 + int(curr_file_count))
                txn.put(value.kmer.encode(), new_file_count.encode(), db=file_counts)

                print_status(counter, inner_total, i+1, outer_total, verbose)
                counter += 1
            print_status(inner_total, inner_total, i+1, outer_total, verbose)

        print('Backfilling missing values in DB')
        for i, db in enumerate(db_keys):
            current = env.open_db(db.encode(), txn=txn)
            with txn.cursor(db=global_counts) as cursor:
                num_kmers = txn.stat(global_counts)['entries']
                counter = 0
                print_status(counter, num_kmers, i, outer_total, verbose)
                for key, value in cursor:
                    if not txn.get(key, default=False, db=current):
                        txn.put(key, '0'.encode(), db=current)
                    print_status(counter, num_kmers, i+1, outer_total, verbose)
                    counter += 1
                print_status(num_kmers, num_kmers, i+1, outer_total, verbose)

        print('Preparing output')
        num_keys = txn.stat(global_counts)['entries']
        for index, value in enumerate(db_keys):
            current = env.open_db(value.encode(), txn=txn)
            output = np.zeros(num_keys, dtype='float64')
            with txn.cursor(db=current) as cursor:
                print_status(0, num_keys, index+1, len(files), verbose)
                for i, v in enumerate(cursor):
                    output[i] = float(v[1])
                    print_status(i, num_keys, index+1, len(files), verbose)
            print_status(num_keys, num_keys, index+1, len(files), verbose)
            txn.put('complete'.encode(), output.tostring(), db=current)

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

def get_counts(k, files, directory):
    output_files, db_keys, database = get_output_files(files, directory, k)
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
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
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)

    global_counts = env.open_db('global_counts'.encode(), create=False)
    with env.begin(write=False, db=global_counts) as txn:
        kmer_list = []
        with txn.cursor(db=global_counts) as cursor:
            for item in cursor:
                kmer_list.append(item[0].decode())
    env.close()
    return np.asarray(kmer_list)





