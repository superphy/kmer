
# from data_augmentation import augment_data_naive, augment_data_smote, augment_data_adasyn
from feature_selection import variance_threshold, select_k_best
from feature_selection import select_percentile, recursive_feature_elimination
from feature_selection import recursive_feature_elimination_cv
from best_models import neural_network, support_vector_machine
from data import get_kmer_us_uk_mixed, get_kmer_us_uk_split
from data import get_genome_region_us_uk_mixed, get_genome_region_us_uk_split
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

data_methods = {'kmer_mixed': get_kmer_us_uk_mixed,
                'kmer_split': get_kmer_us_uk_split,
                'genome_mixed': get_genome_region_us_uk_mixed,
                'genome_split': get_genome_region_us_uk_split}

models = {'nn': neural_network,
          'svm': support_vector_machine}

score_functions = {'f': f_classif,
                   'chi2': chi2,
                   'mutaul_info': mutual_info_classif}

def test_variance():
    sel = variance_threshold
    variances = list(np.arange(0.044,1.0,0.002))
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
            data = get_genome_region_us_uk_split(threeD=True)
            data = sel(*data, threshold=v)
            nn_split_genome.append(neural_network(*data))
            data = get_genome_region_us_uk_split(threeD=False)
            data = sel(*data, threshold=v)
            svm_split_genome.append(support_vector_machine(*data))
            data = get_genome_region_us_uk_mixed(threeD=True)
            data = sel(*data, threshold=v)
            nn_mixed_genome.append(neural_network(*data))
            data = get_genome_region_us_uk_mixed(threeD=False)
            data = sel(*data, threshold=v)
            svm_mixed_genome.append(support_vector_machine(*data))
            data = get_kmer_us_uk_split(threeD=True)
            data = sel(*data, threshold=v)
            nn_split_kmer.append(neural_network(*data))
            data = get_kmer_us_uk_split(threeD=False)
            data = sel(*data, threshold=v)
            svm_split_kmer.append(support_vector_machine(*data))
            data = get_kmer_us_uk_mixed(threeD=True)
            data = sel(*data, threshold=v)
            nn_mixed_kmer.append(neural_network(*data))
            data = get_kmer_us_uk_mixed(threeD=False)
            data = sel(*data, threshold=v)
            svm_mixed_kmer.append(support_vector_machine(*data))
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



# augmentation_methods = {'naive': [augment_data_naive, 2],
#                        'smote': [augment_data_smote, 186],
#                        'adasyn': [augment_data_adasyn, 186]}
#
#
#
#
# def test_feature_selectors():
#     file = '/home/rboothman/Data/feature_selection/results_2.txt'
#     with open(file, 'a') as f:
#         f.write('data,feature_selector,model,score_function,score')
#     for data_key in data_methods:
#         data_method = data_methods[data_key]
#         for sel_key in feature_selectors:
#             feature_selector = feature_selectors[sel_key]
#             if sel_key in ['k', 'percentile']:
#                 for s_func_key in score_functions:
#                     s_func = score_functions[s_func_key]
#                     for model_key in models:
#                         model = models[model_key]
#                         total = 0.0
#                         for i in range(10):
#                             if model_key == 'nn':
#                                 threeD = True
#                             else:
#                                 threeD = False
#                             if s_func_key == 'chi2':
#                                 scale = False
#                             else:
#                                 scale = True
#                             data = data_method(threeD=threeD, scale=scale)
#                             data = feature_selector(*data, score_func=s_func)
#                             total += model(*data)
#                         score = total/10.0
#                         print data_key, sel_key, model_key, s_func_key, score
#                         with open(file, 'a') as f:
#                             f.write('%s,%s,%s,%s:%f'%(data_key,sel_key,model_key,s_func_key,score))
#             else:
#                 for model_key in models:
#                     model = models[model_key]
#                     total = 0.0
#                     for i in range(10):
#                         if model_key == 'nn':
#                             data = data_method()
#                         else:
#                             data = data_method(threeD=False)
#                         data = feature_selector(*data)
#                         total += model(*data)
#                     score = total/10.0
#                     print data_key, sel_key, model_key, 'NA', score
#                     with open(file, 'a') as f:
#                         f.write('%s,%s,%s,NA:%f'%(data_key,sel_key,model_key,score))
#
#
#
# def just_data_augmentation():
#     augmentations = augmentation_methods.keys()
#     ans = {'default':0}
#     maximum = 'default'
#     x_train, y_train, x_test, y_test = get_preprocessed_data("database",True,False,7,13)
#     for augmentation in augmentations:
#         method = augmentation_methods[augmentation][0]
#         param = augmentation_methods[augmentaion][1]
#         x_train, y_train = method(x_train,y_train,param)
#         model = define_model(x_train)
#         model.fit(x_train, y_train, batch_size=10, epochs=120, verbose=1)
#         score = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
#         print augmentation, score[1]
#         ans[augmentation] = score[1]
#         if score[1] >= ans[maximum]:
#             maximum = augmentation
#
#     return maximum, ans[maximum]
#
#
#
# def both_selection_and_augmentation():
#     augmentations = augmentation_methods.keys()
#     selectors = feature_selectors.keys()
#     ans = {'default':0}
#     maximum = 'default'
#     x_train, y_train, x_test, y_test = get_preprocessed_data("database",True,False,7,13)
#     for augmentation in augmentations:
#         for selector in selectors:
#             aug_method = augmentation_methods[augmentation][0]
#             aug_param = augmentation_methods[augmentaion][1]
#             sel_method = feature_selectors[selector][0]
#             sel_param = feature_selectors[selector][1]
#             x_train,y_train,x_test,y_test = sel_method(x_train,y_train,x_test,y_test,sel_param)
#             x_train, y_train = aug_method(x_train, y_train, aug_param)
#             model = define_modle(x_train)
#             model.fit(x_train, y_train, batch_size=10, epochs=120, verbose=1)
#             score = model.evaluate(x_test, y_test, batch_size=10, verbsoe=0)
#             key = augmentation + " " + selector
#             print key, score[1]
#             ans[key] = score[1]
#             if score[1] >= ans[maximum]:
#                 maximum = key
#
#     return maximum, ans[maximum]
#
#
#
# def main():
#     best = just_feature_selection()
#     print "Just Feature Selection: ", best
#     best = just_data_augmentation()
#     print "Just Data Augmentation: ", best
#     best = both_selection_and_augmentation()
#     print "Both Feature Selection and Data Augmentation: ", best


if __name__ == "__main__":
    test_variance()
