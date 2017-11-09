from feature_scaling import scale_to_range
from feature_selection import select_k_best
from models import neural_network_validation, support_vector_machine_validation
from data import get_kmer_us_uk_mixed, get_kmer_us_uk_split
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
from sklearn.feature_selection import chi2, f_classif
import numpy as np
from run import run
from multiprocessing import Process

def get_data(model, data, selection, selection_args, name):
    print name
    path = '/home/rboothman/Data/feature_selection/k_best_f/'
    d = run(model=model,
            data=data,
            selection=selection,
            selection_args=selection_args)
    with open(path+name+'.txt', 'a') as f:
        f.write('%f,%f\n' % (selection_args[1],d))



def test_selection_method(method, score):
    k_vals = list(np.arange(10,1000,10))
    for k in k_vals:
        args = (score, k)
        print "K_val: ", k
        p = Process(target=get_data,
                    args=(neural_network_validation,
                    get_genome_region_us_uk_split,
                    method,
                    args,
                    'nn_split_genome_region'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                    get_genome_region_us_uk_mixed,
                    method,
                    args,
                    'nn_mixed_genome_region'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                    get_kmer_us_uk_mixed,
                    method,
                    args,
                    'nn_mixed_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                    get_kmer_us_uk_split,
                    method,
                    args,
                    'nn_split_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                    get_kmer_us_uk_split,
                    method,
                    args,
                    'svm_split_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                    get_kmer_us_uk_mixed,
                    method,
                    args,
                    'svm_mixed_kmer'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                    get_genome_region_us_uk_mixed,
                    method,
                    args,
                    'svm_mixed_genome_region'))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                          get_genome_region_us_uk_split,
                          method,
                          args,
                          'svm_split_genome_region'))
        p.start()
        p.join()

if __name__ == "__main__":
    test_selection_method(select_k_best, f_classif)
