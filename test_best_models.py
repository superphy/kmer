import numpy as np
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
from data import get_kmer_us_uk_mixed, get_kmer_us_uk_split
import time
from models import support_vector_machine_validation as svm
from models import neural_network_validation as nn
from multiprocessing import Process
from run import run

path = '/home/rboothman/Data/genome_regions/binary_tables/'

def svm_kmer_mixed(reps):
    scores = []
    times = []
    # temp = get_kmer_us_uk_mixed("database", False, True, 7, 13)
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_kmer_us_uk_mixed("database",False,False,7,13)
        scores.append(support_vector_machine_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["svm",
            "kmer counts",
            "mixed",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_svm_kmer_mixed(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=svm,
                                                                data=get_kmer_us_uk_mixed,
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.056),
                                                                reps=reps)
    output = ["svm", "kmer count", "mixed", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def svm_kmer_split(reps):
    scores = []
    times = []
    # temp = get_kmer_us_uk_split("database",False,True,7,13)
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_kmer_us_uk_split("database",False,False,7,13)
        scores.append(support_vector_machine_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["svm",
            "kmer counts",
            "us/uk",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_svm_kmer_split(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=svm,
                                                                data=get_kmer_us_uk_split,
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.164),
                                                                reps=reps)
    output = ["svm", "kmer count", "split", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def svm_genome_mixed(reps):
    scores = []
    times = []
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_genome_region_mixed("binary_table.txt",False)
        scores.append(support_vector_machine_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["svm",
            "genome regions",
            "mixed",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_svm_genome_mixed(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=svm,
                                                                data=get_genome_region_us_uk_mixed,
                                                                data_args={'table':path+'binary_table_450.txt'},
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.008),
                                                                reps=reps)
    output = ["svm", "genome regions", "mixed", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def svm_genome_split(reps):
    scores = []
    times = []
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_genome_region_us_uk_split("binary_table.txt",False)
        scores.append(support_vector_machine_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["svm",
            "genome regions",
            "us/uk",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_svm_genome_split(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=svm,
                                                                data=get_genome_region_us_uk_split,
                                                                data_args={'table':path+'binary_table_1450.txt'},
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.244),
                                                                reps=reps)
    output = ["svm", "genome regions", "split", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def keras_kmer_split(reps):
    scores = []
    times = []
    # temp = get_kmer_us_uk_split("database", True, True, 7, 13)
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_kmer_us_uk_split("database",True,False,7,13)
        scores.append(neural_network_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["neural net",
            "kmer counts",
            "us/uk",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_nn_kmer_split(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=nn,
                                                                data=get_kmer_us_uk_split,
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.152),
                                                                reps=reps)
    output = ["nn", "kmer count", "split", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def keras_kmer_mixed(reps):
    scores = []
    times = []
    # temp = get_kmer_us_uk_mixed("database", True, True, 7, 13)
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_kmer_us_uk_mixed("database",True,False,7,13)
        scores.append(neural_network_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["neural net",
            "kmer counts",
            "mixed",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_nn_kmer_mixed(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=nn,
                                                                data=get_kmer_us_uk_mixed,
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.088),
                                                                reps=reps)
    output = ["nn", "kmer count", "mixed", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def keras_genome_split(reps):
    scores = []
    times = []
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_genome_region_us_uk_split("binary_table.txt",True)
        scores.append(neural_network_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["neural net",
            "genome regions",
            "us/uk",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_nn_genome_split(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=nn,
                                                                data=get_genome_region_us_uk_split,
                                                                data_args={'table':path+'binary_table_1500.txt'},
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.248),
                                                                reps=reps)
    output = ["nn", 'genome regions', "split", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

def keras_genome_mixed(reps):
    scores = []
    times = []
    for i in range(reps):
        start_time = time.time()
        x_train, y_train, x_test, y_test = get_genome_region_mixed("binary_table.txt",True)
        scores.append(neural_network_validation(x_train, y_train, x_test, y_test))
        times.append(time.time()-start_time)
    scores = np.asarray(scores, dtype='float64')
    times = np.asarray(times, dtype='float64')
    output=["neural net",
            "genome regions",
            "mixed",
            scores.mean(),
            scores.std(),
            times.mean(),
            times.std(),
            x_train.shape[0]+x_test.shape[0],
            reps]
    print output
    return ','.join([str(x) for x in output])

def run_nn_genome_mixed(reps):
    acc_mean,acc_std_dev,time_mean,time_std_dev,data_size = run(model=nn,
                                                                data=get_genome_region_us_uk_mixed,
                                                                data_args={'table':path+'binary_table_1500.txt'},
                                                                record_time=True,
                                                                record_std_dev=True,
                                                                record_data_size=True,
                                                                selection_args=(0.008),
                                                                reps=reps)
    output = ["nn", 'genome regions', "mixed", acc_mean, acc_std_dev, time_mean,
              time_std_dev, data_size, reps]
    print output
    return ','.join([str(x) for x in output])

if __name__ == "__main__":
    reps = 10
    skm = run_svm_kmer_mixed(reps)
    sks = run_svm_kmer_split(reps)
    sgm = run_svm_genome_mixed(reps)
    sgs = run_svm_genome_split(reps)
    kkm = run_nn_kmer_mixed(reps)
    kks = run_nn_kmer_split(reps)
    kgm = run_nn_genome_mixed(reps)
    kgs = run_nn_genome_split(reps)

    headers = "Model Type,Input, Train/Test, Acc.Avg., Acc.StdDev, Time Avg., Time StdDev, #Genomes, #Repititions"
    output = [headers, skm, sks, sgm, sgs, kkm, kks, kgm, kgs]
    output = '\n'.join(output)
    with open('/home/rboothman/Documents/best_model_results.csv', 'a') as f:
        currTime = time.strftime("%c")
        f.write('%s\n'%currTime)
        f.write(output)
        f.write('\n')
