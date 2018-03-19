import lmdb
import time
import sys
import os
import subprocess
import numpy as np
import pandas as pd
from threading import Thread

def num_to_kmer(num):
    """
    Convert a base 4 number into a kmer
    """
    out = []
    for d in num:
        if d == '0':
            out.append('A')
        elif d == '1':
            out.append('C')
        elif d == '2':
            out.append('G')
        elif d == '3':
            out.append('T')
    out = ''.join(out)
    return out


def reverse_complement(kmer):
    """
    Return the lexicographically smaller of kmer and it's reverse complement
    """
    reverse = kmer[::-1]
    complement = []
    for bp in reverse:
        if bp == 'A':
            complement.append('T')
        elif bp == 'T':
            complement.append('T')
        elif bp == 'C':
            complement.append('G')
        elif bp == 'G':
            complement.append('C')
    complement = ''.join(complement)
    if complement < kmer:
        out = complement
    else:
        out = kmer
    return out


def random_kmers(k, N):
    """
    Generate N random k-mers of length k
    """
    nums = np.random.randint(0, 4**k, size=N)
    base_4 = [np.base_repr(x, base=4, padding=k)[-k:] for x in nums]
    kmers = [num_to_kmer(x) for x in base_4]
    canonical = [reverse_complement(x) for x in kmers]
    return canonical


def count_file(filename, k, directory):
    """
    Count all k-mers of length k in filename, store .jf and human/python
    interpretable results files in directory
    """
    name = filename.split('/')[-1]
    jellyfishfile = directory + name + '.jf'
    csvfile = directory + name + '.csv'

    # count k-mers for filename
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '32',
            '-C', str(filename), '-o', str(jellyfishfile)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    # convert jellyfish output to tsv
    args = ['jellyfish', 'dump', '-c', '-t', str(jellyfishfile),
            '-o', str(csvfile)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    return csvfile


def make_master_db(kmers, env):
    """
    Add all kmers as keys to db named 'master' in env with values set to 0
    """
    master = env.open_db('master'.encode())
    with env.begin(write=True, db=master) as txn:
        txn.drop(master, delete=False)
        for k in kmers:
            txn.put(k.encode(), '0'.encode())


def add_to_db(filename, kmers, env):
    """
    Add the count of every kmer in kmers from filename to a db in env
    """
    df = pd.read_csv(filename, index_col=0, names=['Count'], sep='\t')
    dictionary = df.to_dict()['Count']

    current = env.open_db(filename.encode())
    master = env.open_db('master'.encode())
    with env.begin(write=True, db=master) as txn:
        for k in kmers:
            if k in dictionary:
                txn.put(k.encode(), '1'.encode(), db=master)
                txn.put(k.encode(), str(dictionary[k]).encode(), db=current)
            else:
                txn.put(k.encode(), '0'.encode(), db=current)


def check_master(env):
    """
    Return the number of keys whose value is still 0 in 'master'
    Add all the keys whose value is still 0 in 'master' to named db 'misses'
    """
    master = env.open_db('master'.encode())
    misses = env.open_db('misses'.encode())
    count = 0
    with env.begin(write=True, db=master) as txn:
        with txn.cursor() as cursor:
            for key, value in cursor:
                if value == '0'.encode():
                    txn.delete(key, db=master)
                    txn.put(key, '0'.encode(), db=misses)
                else:
                    count += 1
    return count


def remove_misses(filename, env):
    """
    Drop every key, value from filename that appears in 'misses'
    """
    misses = env.open_db('misses'.encode())
    current = env.open_db(filename.encode())
    with env.begin(write=True, db=misses) as txn:
        with txn.cursor() as cursor:
            for key, value in cursor:
                txn.delete(key, db=current)


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


def count_kmers(k, count, files, directory, verbose):
    total = len(files)
    directory = directory if directory[-1] == '/' else directory + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    database = directory + 'DB/'

    csvfiles = [directory + x.split('/')[-1] + '.csv' for x in files]
    threads = []
    for f in files:
        threads.append(Thread(target=count_file, args=[f, k, directory]))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)

    kmers = random_kmers(k, 2*count)
    make_master_db(kmers, env)

    status = 0
    for f in csvfiles:
        print_status(status, total, verbose)
        add_to_db(f, kmers, env)
        status += 1
    print_status(status, total, verbose)

    found = check_master(env)

    while found < count:
        kmers = random_kmers(k, count)
        make_master_db(kmers, env)
        status = 0
        for f in csvfiles:
            print_status(status, total, verbose)
            add_to_db(f, kmers, env)
            status += 1
        print_status(status, total, verbose)
        found += check_master(env)

    status = 0
    for f in csvfiles:
        remove_misses(f, env)
        print_status(status, total, verbose)
        status += 1
    print_status(status, total, verbose)

    env.close()


def get_counts(files, directory):
    directory = directory if directory[-1] == '/' else directory + '/'
    database = directory + 'DB/'

    csvfiles = [directory + x.split('/')[-1] + '.csv' for x in files]

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


def get_kmer_names(directory):
    directory = directory if directory[-1] == '/' else directory + '/'
    database = directory + 'DB/'
    env = lmdb.open(str(database), map_size=int(160e9), max_dbs=4000)
    data = env.open_db('master'.encode(), dupsort=False)

    with env.begin(write=False, db=data) as txn:
        kmer_list = []
        with txn.cursor() as cursor:
            for item in cursor:
                kmer_list.append(item[0].decode())
    env.close()
    return np.asarray(kmer_list)
