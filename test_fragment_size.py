from best_models import neural_network, support_vector_machine
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
import matplotlib.pyplot as plt
import numpy as np

fragment_sizes = [(100*(x+1))-50 for x in range(20)]
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
        temp = get_genome_region_us_uk_mixed(table, True)
        nn_mixed.append(neural_network(*temp))
        temp = get_genome_region_us_uk_split(table, True)
        nn_split.append(neural_network(*temp))
        temp = get_genome_region_us_uk_mixed(table, False)
        svm_mixed.append(support_vector_machine(*temp))
        temp = get_genome_region_us_uk_split(table, False)
        svm_split.append(support_vector_machine(*temp))
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

# data = [nn_mixed, nn_split, svm_mixed, svm_split]
# labels = ['Neural Network Mixed', 'Neural Network US/UK Split', 'SVM Mixed', 'SVM US/UK Split']
#
# plt.figure(1)
# for i in range(len(data)):
#     plt.plot(fragment_sizes, data[i], label=labels[i])
# plt.xlabel('Fragmentation Size')
# plt.ylabel('Percent Correct')
# plt.xticks(np.arange(0,1500,100))
# plt.yticks(np.arange(0.3,1.0,0.025))
# plt.grid()
# plt.show()
