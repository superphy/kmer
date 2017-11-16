from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.model_selection import GridSearchCV
from data import get_genome_region_us_uk_split as data
from feature_scaling import scale_to_range as scale
from feature_selection import select_k_best as sel
from sklearn.feature_selection import f_classif
import numpy as np

def kmer_split_sgd(input_data):
    model = SGDC(loss='log', n_jobs=-1, eta0=1.0,
                 learning_rate='invscaling', penalty='none', tol=0.001,
                 alpha=100000000.0)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def kmer_mixed_sgd(input_data):
    model = SGDC(loss='squared_hinge', n_jobs=-1, penalty='none',
                 tol=0.001, alpha=10000000.0)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def genome_split_sgd(input_data):
    model = SGDC(loss='hinge', n_jobs=-1, eta0=0.1,
                 learning_rate='invscaling', penalty='l1', tol=0.001,
                 alpha=0.01)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def genome_mixed_sgd(input_data):
    model = SGDC(loss='log', n_jobs=-1, eta0=0.1,
                 learning_rate='invscaling', penalty='l1', tol=0.001,
                 alpha=0.001)
    model.fit(input_data[0], input_data[1])
    return model.score(input_data[2], input_data[3])

def main():
    reps = 10
    vals = np.zeros(reps)
    for i in range(reps):
        d = data()
        # d = sel(d, *(f_classif, 270))
        # d = scale(d, *(-1,1))
        score = gmixed_model(d)
        print score
        vals[i] = score
    print vals.mean()

if __name__ == "__main__":
    main()
