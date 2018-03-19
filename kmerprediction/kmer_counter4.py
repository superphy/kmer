import lmdb
import time
import sys
import os
import subprocess
import numpy as np
import pandas as pd
from threading import Thread
import tempfile

def print_status(status, total):
    p = int((status*100)/total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-44s] %d%%" % ('='*int((p*44)/100), p))
    sys.stdout.flush()
    if p == 100:
        print("\n")

def count_file(input_file, output_file, k):
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

def add_file_to_master(filename, max_add, env):
    master = env.open_db('master'.encode())
    with env.begin(write=True, db=master) as txn:
        data = pd.read_csv(filename, names=['Kmer', 'Count'], sep='\t')
        randx = np.random.randint(data.shape[0], size=data.shape[0])
        count = 0
        for x in randx:
            if count >= max_add:
                break
            kmer = data.loc[x]['Kmer'].encode()
            if not txn.get(kmer, default=False, db=master):
                txn.put(kmer, '1'.encode(), db=master)
                count += 1
    return count

def make_master_db(files, N, env):
    count = 0
    kmers_per_file = int(np.ceil(N/len(files)))
    for f in files:
        if count >= N:
            break
        kmers_added = add_file_to_master(f, kmers_per_file, env)
        print_status(count, N)
        count += kmers_added
        if kmers_added == 0:
            break
    print_status(N, N)

def add_to_db(filename, env):
    df = pd.read_csv(filename, index_col=0, names=['Count'], sep='\t')
    dictionary = df.to_dict()['Count']

    current = env.open_db(filename.encode())
    master = env.open_db('master'.encode())
    with env.begin(write=True, db=master) as txn:
        txn.drop(current, delete=False)
        with txn.cursor() as cursor:
            for key, value in cursor:
                if key.decode() in dictionary:
                    txn.put(key, str(dictionary[key.decode()]).encode(), db=current)
                else:
                    txn.put(key, '0'.encode(), db=current)

def count_kmers(k, N, files, directory, verbose):
    if not os.path.exists(directory):
        os.makedirs(directory)

    max_kmers = int((4**k)/2)
    N = N if N < max_kmers else max_kmers

    database = directory + 'k{}_N{}.db'.format(k, N)
    csvfiles = [directory + x.split('/')[-1].split('.')[0] + '_k{}'.format(k) for x in files]

    threads = []
    for i, f in enumerate(files):
        threads.append(Thread(target=count_file, args=(f, csvfiles[i], k)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    env = lmdb.open(database, map_size=int(160e9), max_dbs=4000)

    master = env.open_db('master'.encode())
    with env.begin(write=True, db=master) as txn:
        txn.drop(master, delete=True)

    make_master_db(csvfiles, N, env)

    status = 0
    total = len(csvfiles)
    for f in csvfiles:
        add_to_db(f, env)
        print_status(status, total)
        status += 1
    print_status(status, total)

    env.close()

def get_counts(k, N, files, directory):
    max_kmers = int((4**k)/2)
    N = N if N < max_kmers else max_kmers
    database = directory + 'k{}_N{}.db'.format(k, N)
    csvfiles = [directory + x.split('/')[-1].split('.')[0] + '_k{}'.format(k) for x in files]

    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)

    master = env.open_db('master'.encode(), dupsort=False)
    with env.begin(write=False, db=master) as txn:
        if csvfiles:
            first = env.open_db(csvfiles[0].encode(), txn=txn)
        else:
            return np.array([], dtype='float64')
        num_keys = txn.stat(first)['entries']
        output = np.zeros((len(files), num_keys), dtype='float64')

        for index, value in enumerate(csvfiles):
            current = env.open_db(value.encode(), txn=txn)
            with txn.cursor(db=current) as cursor:
                counter = 0
                for item in cursor:
                    output[index, counter] = float(item[1])
                    counter += 1
    env.close()
    return output


def get_kmer_names(k, N, directory):
    max_kmers = int((4**k)/2)
    N = N if N < max_kmers else max_kmers

    database = directory + 'k{}_N{}.db'.format(k, N)

    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)

    data = env.open_db('master'.encode(), dupsort=False)
    with env.begin(write=False, db=data) as txn:
        kmer_list = []
        with txn.cursor() as cursor:
            for item in cursor:
                kmer_list.append(item[0].decode())
    env.close()
    return np.asarray(kmer_list)

