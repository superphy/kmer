from feature_scaling import scale_to_range
from feature_selection import recursive_feature_elimination
from models import neural_network_validation, support_vector_machine_validation
from data import get_kmer_us_uk_mixed, get_kmer_us_uk_split
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
from sklearn.feature_selection import chi2, f_classif
import numpy as np
from run import run
from multiprocessing import Process
from sklearn.svm import SVC
from data_augmentation import augment_data_adasyn

def get_data(model, data, augment, k, name):
    print name
    path = '/home/rboothman/Data/data_augmentation/adasyn/'
    selection_args = (k, 0, 1)
    d = run(model=model,
            data=data,
            augment=augment,
            augment_args=selection_args)
    with open(path+name+'.txt', 'a') as f:
        f.write('%f,%f\n' % (k,d))

def test_selection_method(method):
    k_vals = list(np.arange(5,250,5))
    for k in k_vals:
        print "K_val: ", k
        p = Process(target=get_data,
                    args=(neural_network_validation,
                          get_genome_region_us_uk_split,
                          method,
                          k,
                          'nn_split_genome_region'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                           get_genome_region_us_uk_mixed,
                           method,
                           k,
                           'nn_mixed_genome_region'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                          get_kmer_us_uk_mixed,
                          method,
                          k,
                          'nn_mixed_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                          get_kmer_us_uk_split,
                          method,
                          k,
                          'nn_split_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                          get_kmer_us_uk_split,
                          method,
                          k,
                          'svm_split_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                          get_kmer_us_uk_mixed,
                          method,
                          k,
                          'svm_mixed_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                          get_genome_region_us_uk_mixed,
                          method,
                          k,
                          'svm_mixed_genome_region'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                          get_genome_region_us_uk_split,
                          method,
                          k,
                          'svm_split_genome_region'))
        p.start()
        p.join()

if __name__ == "__main__":
    test_selection_method(augment_data_adasyn)
