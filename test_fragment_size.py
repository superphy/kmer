from models import neural_network_validation, support_vector_machine_validation
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
import numpy as np

fragment_sizes = [50*(x+1) for x in range(20)]
data_path = '/home/rboothman/Data/genome_regions/binary_tables/'

for fragment_size in fragment_sizes:
    nn_mixed = []
    nn_split = []
    svm_mixed = []
    svm_split = []
    print "Fragment Size: %d" % fragment_size
    for i in range(10):
        print "Repitition: %d" % i
        table = data_path + 'binary_table_%d.txt'%fragment_size
        data = get_genome_region_us_uk_mixed(table)
        nn_mixed.append(neural_network_validation(*data))
        svm_mixed.append(support_vector_machine_validation(*data))
        data = get_genome_region_us_uk_split(table)
        nn_split.append(neural_network_validation(*data))
        svm_split.append(support_vector_machine_validation(*data))
    nn_mixed = np.asarray(nn_mixed, dtype='float64').mean()
    nn_split = np.asarray(nn_split, dtype='float64').mean()
    svm_mixed = np.asarray(svm_mixed, dtype='float64').mean()
    svm_split = np.asarray(svm_split, dtype='float64').mean()
    print nn_mixed, nn_split, svm_mixed, svm_split
    with open('/home/rboothman/Data/genome_regions/nn_mixed_50.txt', 'a') as f:
        f.write('%d,%f\n' % (fragment_size, nn_mixed))
    with open('/home/rboothman/Data/genome_regions/nn_split_50.txt', 'a') as f:
        f.write('%d,%f\n' % (fragment_size, nn_split))
    with open('/home/rboothman/Data/genome_regions/svm_mixed_50.txt', 'a') as f:
        f.write('%d,%f\n' % (fragment_size, svm_mixed))
    with open('/home/rboothman/Data/genome_regions/svm_split_50.txt', 'a') as f:
        f.write('%d,%f\n' % (fragment_size, svm_split))
