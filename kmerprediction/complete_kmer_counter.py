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
import logging

class KmerCounterError(Exception):
    """Raise for errors in kmer_counter module"""


def count_file(input_file, output_file, k):
    """
    Use jellyfish to count kmers of length k in input_file and store the
    result in output_file.

    Args:
        input_file (str):   Path to a fasta file to count kmers in.
        output_file (str):  Path to a csv file to store results in.
        k (int)             Length of kmer to count.

    Returns:
        None
    """
    handle, temp_file = tempfile.mkstemp()
    args = ['jellyfish', 'count', '-m', '%d' % k, '-s', '10M', '-t', '30',
            '-C', str(input_file), '-o', str(temp_file)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    args = ['jellyfish', 'dump', '-c', '-t', str(temp_file), '-o', str(output_file)]
    p = subprocess.Popen(args, bufsize=-1)
    p.communicate()

    os.remove(temp_file)
    logging.info('Counted kmers for {}'.format(input_file))


def count_all(fasta_files, temp_files, db_keys, k, env, force):
    """
    Counts kmers of length k for each fasta file in fasta_files in parrallel.

    Args:
        fasta_files (list):     Paths to fasta files to count kmers in.
        temp_files (list):      Paths to files to store the csv jellyfish
                                output in.
        db_keys (list):         The lmdb keys to identify each file in the
                                database with.
        k (int):                The length of kmer to count.
        env (lmdb.Envrionment): Environment containing the database to store
                                the complete results in.
        force (bool):           If True kmers for all files are recounted, if
                                False only Files that do not appear in the
                                database already are recounted.
    Returns:
        recount (list): Every db_key whose kmers were recounted.
    """
    logging.info('Begin Counting kmers')
    threads = []
    recounts = []
    if force:
        logging.info('Force set to True, recounting all genomes')
    for i, v in enumerate(fasta_files):
        with env.begin(write=False) as txn:
            if force or not txn.get(db_keys[i].encode(), default=False):
                args = [v, temp_files[i], k]
                threads.append(Thread(target=count_file, args=args))
                recounts.append(db_keys[i])
    if not force:
        logging.info('Force set to False, Recounting {}'.format(recounts))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info('Done counting kmers')
    return recounts


def add_file(input_file, key, env, global_counts, file_counts):
    """
    Add the kmer counts contained in the input csv file to the database in
    env under the identifier key. Update global_counts and file_counts.

    Args:
        input_file (str):           Path to a csv file containing jellyfish
                                    output.
        key (str):                  Identifier for a named database
                                    corresponding to input_file
        env (lmdb.Environment):     Environment containing the database to store
                                    the complete results in.
        global_counts (database):   Named database with keys of kmers and values
                                    of their total count across all files in the
                                    database.
        file_counts (database):     Named database with keys of kmers and values
                                    of the number of files they appear in across
                                    the database.
    Returns:
        None
    """
    current = env.open_db(key.encode())
    with env.begin(write=True, db=current) as txn:
        with open(input_file, 'r') as f:
            for line in f:
                kmer = line.split()[0].encode()
                count = str(line.split()[1]).encode()
                txn.put(kmer, count, db=current)

                curr_global_count = txn.get(kmer, default=0, db=global_counts)
                new_global_count = str(int(curr_global_count) + int(count)).encode()
                txn.put(kmer, new_global_count, db=global_counts)

                curr_file_count = txn.get(kmer, default=0, db=file_counts)
                new_file_count = str(1 + int(curr_file_count)).encode()
                txn.put(kmer, new_file_count, db=file_counts)
    logging.info('Added {} to DB'.format(key))


def add_all(temp_files, db_keys, global_counts, file_counts, env, force, recounts):
    """
    Add all kmer counts from each file in temp_files to the database
    contained in env under the identifiers contianed in db_keys.

    Args:
        temp_files (list):      The paths to each csv file output by jellyfish.
        db_keys (list):         The identifiers corresponding to each file in
                                temp_files to be used as names in the database.
        env (lmdb.Environment): Environment containing the database to store
                                the complete results in.
        force (bool):           If True every file in temp_file is added to the
                                database, if False only files whose db_key is
                                not present in the database or whose db_key is
                                in recounts are added to the database.
        recounts (list):        A list of all db_keys that were changed by
                                count_all
    Returns:
        recounts (list): Every db_key that was altered in the database.
    """
    logging.info('Begin adding genomes to database')
    threads = []
    if force:
        logging.info('Force set to True, adding all genomes to database')
    for i, f in enumerate(temp_files):
        with env.begin(write=False) as txn:
            if force or not txn.get(db_keys[i].encode(), default=False) or db_keys[i] in recounts:
                args = [f, db_keys[i], env, global_counts, file_counts]
                threads.append(Thread(target=add_file, args=args))
                if db_keys[i] not in recounts:
                    recounts.append(db_keys[i])
    if not force:
        logging.info('Force set to False, Adding {} to the DB'.format(recounts))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info('Done adding genomes to database')
    return recounts


def backfill_file(db_key, env, global_counts):
    """
    Insert all kmers into db_key with a value of 0 that appear in the database,
    but not in db_key.

    Args:
        db_key (str):               Named database
        env (lmdb.Environment):     Environment containing the database to store
                                    the complete results in.
        global_counts (database):   Named database containing every kmer in the
                                    database with their total count over all
                                    files in the database.
    Returns:
        None
    """
    current = env.open_db(db_key.encode())
    with env.begin(write=True, db=current) as txn:
        with txn.cursor(db=global_counts) as cursor:
            for key, value in cursor:
                if not txn.get(key, default=False, db=current):
                    txn.put(key, '0'.encode(), db=current)
    logging.info('Backfilled {}'.format(db_key))


def backfill_all(db_keys, global_counts, env, force, recounts):
    """
    Backfill every named database in db_keys in parrallel.

    Args:
        db_keys (list):             The identifiers corresponding to each file
                                    in temp_files to be used as names in the
                                    database.
        global_counts (database):   Named database containing every kmer in the
                                    database with their total count over all
                                    files in the database.
        env (lmdb.Environment):     Environment containing the database to store
                                    the complete results in.
        force (bool):               If True every file in temp_file is added to
                                    the database, if False only files whose
                                    db_key is not present in the database or
                                    whose db_key is in recounts are added to the
                                    database.
        recounts (list):            A list of all db_keys that were changed by
                                    count_all.
    Returns:
        recounts (list): Every db_key that was altered in the database.
    """
    logging.info('Begin backfilling genomes')
    with env.begin(write=False, db=global_counts) as txn:
        total_kmers = txn.stat()['entries']
    threads = []
    if force:
        logging.info('Force set to True, backfilling all genomes')
    for k in db_keys:
        curr_db = env.open_db(k.encode())
        with env.begin(write=False, db=curr_db) as txn:
            curr_kmers = txn.stat()['entries']
        if force or curr_kmers < total_kmers or k in recounts:
            args = [k, env, global_counts]
            threads.append(Thread(target=backfill_file, args=args))
            if k not in recounts:
                recounts.append(k)
    if not force:
        logging.info('Force set to False, Backfilling {}'.format(recounts))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info('Done backfilling genomes')
    return recounts


def filter_kmers(global_counts, file_counts, env, max_global_count,
                 min_global_count, max_file_count, min_file_count):
    """
    Get a list of all kmers in the database that appear in more files than
    min_file_count, fewer files than max_file_count, as well as more more
    times than min_global_count in total and fewer times than max_global_count.

    Args:
        global_counts (database):   Database containing counts of how many
                                    times in total each kmer appears in the
                                    database.
        file_counts (database):     Database containing counts of how many
                                    files each kmer appears in.
        env (lmdb.Environment):     Environment containing the database to store
                                    the complete results in.
        min_global_count (int): The minimum number of a times a kemr must appear
                                in the entire database in order to be output.
        max_global_count (int): The maximum number of times a kmer can appear
                                in the entire database in order to be output.
        min_file_count (int):   The minimum number a fasta files that a kmer
                                must appear in inorder to be output.
        max_file_count (int):   The maximum number of fasta files that a kmer
                                can appear in inorder to be outoput.

    Returns:
        valid_kmers (list): Every kmer that appears in the database and meets
                            the requirements.
    """
    global_kmers = []
    valid_kmers = []
    with env.begin(write=False, db=global_counts) as global_txn:
        with global_txn.cursor() as global_cursor:
            for key, global_value in global_cursor:
                global_value = int(global_value)
                if max_global_count:
                    if global_value <= max_global_count and global_value >= min_global_count:
                        global_kmers.append(key)
                else:
                    if global_value >= min_global_count:
                        global_kmers.append(key)
    with env.begin(write=False, db=file_counts) as file_txn:
        for kmer in global_kmers:
            file_value = file_txn.get(kmer)
            if int(file_value) <= max_file_count and int(file_value) >= min_file_count:
                valid_kmers.append(kmer.decode())

    return valid_kmers


def output_file(key, valid_kmers, input_env, output_env, name, write_kmers):
    """
    Convert the key, value pairs in input_env under key to a numpy string
    representation of a 1D array in output_env under key[name].

    Args:
        key (str):                      Identifier for the named database in
                                        both envs.
        valid_kmers (list):             The list of kmers to include in the
                                        output.
        input_env (lmdb.Environment):   Environment containing the complete
                                        kmer count results.
        output_env (lmdb.Environment):  Environment where you want to store
                                        the output values, can be the same
                                        environment as input_env.
        name (str):                     The key to store the output value
                                        under. Usefull when using multiple
                                        kmer filter methods.
        write_kmers (bool):             If True the kmer names are written as
                                        to the output_env under a database
                                        named name.
    Returns:
        None
    """
    db = input_env.open_db(key.encode())
    output = np.zeros(len(valid_kmers), dtype=int)
    kmer_name_db = output_env.open_db(name.encode())
    with input_env.begin(write=False, db=db) as txn_in:
        with output_env.begin(write=True, db=kmer_name_db) as txn_out:
            for index, kmer in enumerate(valid_kmers):
                output[index] = int(txn_in.get(kmer.encode(), db=db))
                if write_kmers:
                    txn_out.put(kmer.encode(), '1'.encode())
    db = output_env.open_db(key.encode())
    with output_env.begin(write=True, db=db) as txn:
        txn.put(name.encode(), output.tostring(), db=db)
    logging.info('Made output key for {}'.format(key))


def output_all(db_keys, valid_kmers, env, output_env, name, force, recounts):
    """
    Create the output value for each key in db_keys in parrallel.

    Args:
        db_keys (list):                 Every db_key to make the output for.
        valid_kmers (list):             The list of kmers to include in the
                                        output.
        env: (limdb.Environment):       Environment containing the complete
                                        kmer count results.
        output_env (lmdb.Environment):  Environment where you want to store the
                                        output values, can be the same
                                        environment as env.
        name (str):                     The key to store the output value under.
                                        Usefull when using multiple kmer filter
                                        methods.
        force (bool):                   If True every output is remade, if False
                                        only db_keys that do not have a valid
                                        name key in output_env or appear in
                                        recounts are remade.
        recounts (list):                Every db_key altered by count_all.
    Returns:
        None
    """
    logging.info('Begin making output')
    threads = []
    if force:
        logging.info('Force set to True, creating all outputs')
    write_kmers = True
    for k in db_keys:
        curr_db = env.open_db(k.encode())
        with env.begin(write=False, db=curr_db) as txn:
            if force or not txn.get(k.encode(), default=False, db=curr_db) or k in recounts:
                args = [k, valid_kmers, env, output_env, name, write_kmers]
                threads.append(Thread(target=output_file, args=args))
                write_kmers = False
                if k not in recounts:
                    recounts.append(k)
    if not force:
        logging.info('Force set to False, creating outputs for {}'.format(recounts))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info('Done making output')


def make_db_keys(input_files):
    """
    Convert a list of files into keys to use in a database

    Args:
        input_files (list): The files to convert.
    Returns:
        output (list): Every input file path with the absolute path and file
                       suffix stripped.
    """
    output = [x.split('/')[-1] for x in input_files]
    output = [x.split('.')[:-1] for x in output]
    output = ['.'.join(x) for x in output]
    return output


def count_kmers(fasta_files, database, k=constants.DEFAULT_K, verbose=True,
                output_db=None, min_global_count=0, max_global_count=None,
                min_file_count=0, max_file_count=None, force=False,
                name=constants.DEFAULT_NAME):
    """
    Count kmers in fasta_files of length k. Store the complete results in
    database and the simplified output in output_db.

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
    logging.info('Begin complete_kmer_counter.count_kmers')
    max_file_count = max_file_count or len(fasta_files) + 1

    db_keys = make_db_keys(fasta_files)
    temp_dir = tempfile.mkdtemp()
    temp_files = [temp_dir + '/' + x for x in db_keys]

    env = lmdb.open(database, map_size=160e10, max_dbs=4000, max_readers=1e7)
    global_counts = env.open_db('global_counts'.encode())
    file_counts = env.open_db('file_counts'.encode())

    recounts = count_all(fasta_files, temp_files, db_keys, k, env, force)
    recounts = add_all(temp_files, db_keys, global_counts, file_counts, env, force, recounts)
    recounts = backfill_all(db_keys, global_counts, env, force, recounts)

    if output_db:
        output_env = lmdb.open(output_db, map_size=160e10, max_dbs=4000,
                               max_readers=1e7)
    else:
        output_env = env

    valid_kmers = filter_kmers(global_counts, file_counts, env,
                               max_global_count, min_global_count,
                               max_file_count, min_file_count)

    output_all(db_keys, valid_kmers, env, output_env, name, force, recounts)

    shutil.rmtree(temp_dir)

    env.close()
    logging.info('Done complete_kmer_counter.count_kmers')


def get_counts(files, database, name=constants.DEFAULT_NAME):
    """
    Get the kmer counts for files stored in database under name.

    Args:
        files (list):   The file to get the counts for.
        database (str): File path to database.
        name (str):     Identifier for the output in database.

    Returns:
        output (ndarray):   An (n_samples, n_features) shape numpy array ready
                            to be passed to a machine learning method.
    """
    db_keys = make_db_keys(files)

    if not os.path.exists(database):
        msg = 'Attempted to get counts from an uncreated database: {}'.format(database)
        raise(KmerCounterError(msg))

    env = lmdb.open(database, map_size=160e10, max_dbs=4000, max_readers=1e7)
    with env.begin(write=False) as txn:
        arrays = []
        for index, value in enumerate(db_keys):

            try:
                current = env.open_db(value.encode(), txn=txn, create=False)
            except lmdb.NotFoundError:
                msg = 'Attempted to get counts for potentially uncounted genome:'
                msg += ' {} in DB: {}'.format(value, database)
                logging.exception(msg)
                raise(KmerCounterError(msg))

            results = txn.get(name.encode(), default=None, db=current)
            if results is None:
                msg = 'Attempted to get counts for potentially invalid filter method:'
                msg += ' {} for genome: {} in DB: {}'.format(name, value, database)
                raise(KmerCounterError(msg))

            results = np.fromstring(results, dtype='int')
            arrays.append(results)
        output = np.vstack(arrays)
    env.close()
    return output


def get_kmer_names(database, name=constants.DEFAULT_NAME):
    """
    Get the names of every kmer in the database.

    Args:
        database (str): Filepath to the database.
        name (str):     Identifier for the output in database.

    Returns:
        output (ndarray):   A (n_features,) shape numpy array containing the names of
                            every kmer in the output.
    """
    env = lmdb.open(database, map_size=160e10, max_dbs=4000, max_readers=1e7)

    try:
        db = env.open_db(name.encode(), create=False)
    except lmdb.NotFoundError:
        msg = 'Attempted to get kmer names from a potentially uncreated'
        msg += ' database: {} in {}'.format(name, database)
        logging.exception(msg)
        raise(KmerCounterError(msg))

    with env.begin(write=False, db=db) as txn:
        output = np.zeros(txn.stat()['entries'], dtype='<U64')
        with txn.cursor() as cursor:
            for index, (key, value) in enumerate(cursor):
                output[index] = key.decode()
    env.close()
    return output


