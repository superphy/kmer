from data import get_genome_region_filteredB as data
from models import support_vector_machine_validation as svm
from models import neural_network_validation as nn
import numpy as np
from multiprocessing import Process

def split(count, model, name, path):
    output = []
    print name
    for i in range(10):
        args = ('Data/human_bovine.csv','Human','Bovine','','','Train','Test')
        output.append(model(*data(args=args, count=count)))
    output = np.asarray(output, dtype='float64')
    mean = output.mean()
    with open(path + name + '.txt', 'a') as f:
        f.write('%f,%f,%s\n' % (count, mean, ','.join([str(x) for x in list(output)])))

def mixed(count, model, name, path):
    output = []
    print name
    for i in range(10):
        args = ('Data/human_bovine.csv', 'Human', 'Bovine')
        output.append(model(*data(args=args, count=count)))
    output = np.asarray(output, dtype='float64')
    mean = output.mean()
    with open(path + name + '.txt', 'a') as f:
        f.write('%f,%f,%s\n' % (count, mean, ','.join([str(x) for x in list(output)])))

if __name__ == "__main__":
    path = '/home/rboothman/Data/feature_selection/genome_filtered/'
    for count in list(np.arange(1000,1510,10)):
        print "Count: ", count
        p = Process(target=split, args=(count, svm, 'split_svm', path))
        p.start()
        p.join()

        p = Process(target=split, args=(count, nn, 'split_nn', path))
        p.start()
        p.join()

        p = Process(target=mixed, args=(count, nn, 'mixed_nn', path))
        p.start()
        p.join()

        p = Process(target=mixed, args=(count, svm, 'mixed_svm', path))
        p.start()
        p.join()
