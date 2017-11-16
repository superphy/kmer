from feature_scaling import scale_to_range
from feature_selection import recursive_feature_elimination, select_k_best
from models import neural_network_validation, support_vector_machine_validation
from data import get_kmer_us_uk_mixed, get_kmer_us_uk_split, get_salmonella_kmer
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
from sklearn.feature_selection import chi2, f_classif
import numpy as np
from run import run
from multiprocessing import Process
from sklearn.svm import SVC

def get_data(model, data, data_args, selection, selection_args, name, path):
    print name
    k = selection_args[1]
    d = run(model=model,
            data=data,
            data_args=data_args,
            selection=selection,
            selection_args=selection_args)
    with open(path+name+'.txt', 'a') as f:
        f.write('%f,%f\n' % (k,d))

def test_selection_method(sel_method, score_func, path, antibiotic):
    k_vals = np.arange(10,1000,10)

    metadata_args = {'antibiotic': antibiotic}
    data_args = ('database2', False, 7, 13, metadata_args)

    for k in k_vals:
        sel_args = (score_func, k)
        print "K: ",k
        # p = Process(target=get_data,
        #             args=(neural_network_validation,
        #                   get_genome_region_us_uk_split,
        #                   args,
        #                   'nn_split_genome_region'))
        # p.start()
        # p.join()
        #
        # p = Process(target=get_data,
        #             args=(neural_network_validation,
        #                    get_genome_region_us_uk_mixed,
        #                    args,
        #                    'nn_mixed_genome_region'))
        # p.start()
        # p.join()
        #
        # p = Process(target=get_data,
        #             args=(neural_network_validation,
        #                   get_kmer_us_uk_mixed,
        #                   args,
        #                   'nn_mixed_kmer'))
        # p.start()
        # p.join()

        p = Process(target=get_data,
                    args=(neural_network_validation,
                          get_salmonella_kmer,
                          data_args,
                          sel_method,
                          sel_args,
                          'neural_net',
                          path))
        p.start()
        p.join()

        p = Process(target=get_data,
                    args=(support_vector_machine_validation,
                          get_salmonella_kmer,
                          data_args,
                          sel_method,
                          sel_args,
                          'svm',
                          path))
        p.start()
        p.join()

        # p = Process(target=get_data,
        #             args=(support_vector_machine_validation,
        #                   get_kmer_us_uk_mixed,
        #                   args,
        #                   'svm_mixed_kmer',
        #                   path))
        # p.start()
        # p.join()
        #
        # p = Process(target=get_data,
        #             args=(support_vector_machine_validation,
        #                   get_genome_region_us_uk_mixed,
        #                   args,
        #                   'svm_mixed_genome_region',
        #                   path))
        # p.start()
        # p.join()
        #
        # p = Process(target=get_data,
        #             args=(support_vector_machine_validation,
        #                   get_genome_region_us_uk_split,
        #                   args,
        #                   'svm_split_genome_region',
        #                   path))
        # p.start()
        # p.join()

if __name__ == "__main__":
    antibiotics = ['ampicillin', 'chloramphenicol', 'gentamicin',
                   'kanamycin', 'nalidixic acid', 'spectinomycin',
                   'streptomycin', 'sulphonamides', 'tetracycline',
                   'trimethoprim']
    base_path = '/home/rboothman/Data/feat_sel_salmonella/f_test/'
    for antibiotic in antibiotics:
        path = base_path + antibiotic + '/'
        test_selection_method(select_k_best, f_classif, path, antibiotic)
