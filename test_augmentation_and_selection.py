from feature_scaling import scale_to_range
from feature_selection import variance_threshold, select_k_best
from feature_selection import select_percentile, recursive_feature_elimination
from feature_selection import recursive_feature_elimination_cv
from models import neural_network_validation, support_vector_machine_validation
from *data import get_kmer_us_uk_mixed, get_kmer_us_uk_split
from *data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
import numpy as np

feature_selectors = {'variance': variance_threshold,
                     'k': select_k_best,
                     'percentile': select_percentile,
                     'recursive': recursive_feature_elimination,
                     'recursive_cv': recursive_feature_elimination_cv}

variances = np.arange(0.0,0.99,0.001)
k = np.arange(10,3000,10)
percentile = np.arange(0.0,0.99,0.001)

*data_methods = {'kmer_mixed': get_kmer_us_uk_mixed,
                'kmer_split': get_kmer_us_uk_split,
                'genome_mixed': get_genome_region_us_uk_mixed,
                'genome_split': get_genome_region_us_uk_split}

models = {'nn': neural_network_validation,
          'svm': support_vector_machine_validation}

score_functions = {'f': f_classif,
                   'chi2': chi2,
                   'mutaul_info': mutual_info_classif}

def test_variance():
    sel = variance_threshold
    variances = list(np.arange(0.134,1.0,0.002))
    for v in variances:
        nn_mixed_kmer = []
        nn_split_kmer = []
        svm_mixed_kmer = []
        svm_split_kmer = []
        nn_mixed_genome = []
        nn_split_genome = []
        svm_mixed_genome = []
        svm_split_genome = []
        print "Variance: ", v
        for i in range(10):
            print "Repitition: ", i
            data = get_genome_region_us_uk_split()
            data = sel(*data, args=[v])
            nn_split_genome.append(neural_network_validation(*data))
            data = get_genome_region_us_uk_split()
            data = sel(*data, args=[v])
            svm_split_genome.append(support_vector_machine_validation(*data))
            data = get_genome_region_us_uk_mixed()
            data = sel(*data, args=[v])
            nn_mixed_genome.append(neural_network_validation(*data))
            data = get_genome_region_us_uk_mixed()
            data = sel(*data, args=[v])
            svm_mixed_genome.append(support_vector_machine_validation(*data))
            data = get_kmer_us_uk_split()
            data = scale_to_range(*data, args=[-1,1])
            data = sel(*data, args=[v])
            nn_split_kmer.append(neural_network_validation(*data))
            data = get_kmer_us_uk_split()
            data = scale_to_range(*data, args=[-1,1])
            data = sel(*data, args=[v])
            svm_split_kmer.append(support_vector_machine_validation(*data))
            data = get_kmer_us_uk_mixed()
            data = scale_to_range(*data, args=[-1,1])
            data = sel(*data, args=[v])
            nn_mixed_kmer.append(neural_network_validation(*data))
            data = get_kmer_us_uk_mixed()
            data = scale_to_range(*data, args=[-1,1])
            data = sel(*data, args=[v])
            svm_mixed_kmer.append(support_vector_machine_validation(*data))
        nn_mixed_kmer = np.asarray(nn_mixed_kmer,dtype='float64')
        nn_mixed_kmer_mean =nn_mixed_kmer.mean()
        nn_split_kmer = np.asarray(nn_split_kmer,dtype='float64')
        nn_split_kmer_mean =nn_split_kmer.mean()
        svm_mixed_kmer = np.asarray(svm_mixed_kmer,dtype='float64')
        svm_mixed_kmer_mean =svm_mixed_kmer.mean()
        svm_split_kmer = np.asarray(svm_split_kmer,dtype='float64')
        svm_split_kmer_mean = svm_split_kmer.mean()
        nn_mixed_genome = np.asarray(nn_mixed_genome,dtype='float64')
        nn_mixed_genome_mean = nn_mixed_genome.mean()
        nn_split_genome = np.asarray(nn_split_genome,dtype='float64')
        nn_split_genome_mean = nn_split_genome.mean()
        svm_mixed_genome = np.asarray(svm_mixed_genome,dtype='float64')
        svm_mixed_genome_mean = svm_mixed_genome.mean()
        svm_split_genome = np.asarray(svm_split_genome,dtype='float64')
        svm_split_genome_mean = svm_split_genome.mean()
        with open('/home/rboothman/Data/feature_selection/variance_threshold/nn_mixed_kmer.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, nn_mixed_kmer_mean, ','.join([str(x) for x in list(nn_mixed_kmer)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/nn_split_kmer.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, nn_split_kmer_mean, ','.join([str(x) for x in list(nn_split_kmer)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/svm_mixed_kmer.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, svm_mixed_kmer_mean, ','.join([str(x) for x in list(svm_mixed_kmer)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/svm_split_kmer.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, svm_split_kmer_mean, ','.join([str(x) for x in list(svm_split_kmer)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/nn_mixed_genome_region.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, nn_mixed_genome_mean, ','.join([str(x) for x in list(nn_mixed_genome)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/nn_split_genome_region.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, nn_split_genome_mean, ','.join([str(x) for x in list(nn_split_genome)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/svm_mixed_genome_region.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, svm_mixed_genome_mean, ','.join([str(x) for x in list(svm_mixed_genome)])))
        with open('/home/rboothman/Data/feature_selection/variance_threshold/svm_split_genome_region.txt', 'a') as f:
            f.write('%f,%f,%s\n' % (v, svm_split_genome_mean, ','.join([str(x) for x in list(svm_split_genome)])))

if __name__ == "__main__":
    test_variance()
