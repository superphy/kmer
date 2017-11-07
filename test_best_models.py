import numpy as np
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
from data import get_kmer_us_uk_mixed, get_kmer_us_uk_split
import time
from models import support_vector_machine_validation, neural_network_validation

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

if __name__ == "__main__":
    reps = 20
    skm = svm_kmer_mixed(reps)
    sks = svm_kmer_split(reps)
    sgm = svm_genome_mixed(reps)
    sgs = svm_genome_split(reps)
    kkm = keras_kmer_mixed(reps)
    kks = keras_kmer_split(reps)
    kgm = keras_genome_mixed(reps)
    kgs = keras_genome_split(reps)

    headers = "Model Type,Input,Train/Test,Avg. Acc.,StdDev Acc.,Avg. Time,StdDev Time,#Genomes,#Repititions"
    output = [headers, skm, sks, sgm, sgs, kkm, kks, kgm, kgs]
    output = '\n'.join(output)
    with open('/home/rboothman/Documents/best_model_results.csv', 'a') as f:
        currTime = time.strftime("%c")
        f.write('%s\n'%currTime)
        f.write(output)
        f.write('\n')
