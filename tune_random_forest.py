from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from data import get_omnilog_data
import constants
import json
import numpy as np


n_estimators_range = [5,10,50,100]
criterion_range = ['gini', 'entropy']
max_features_range = ['auto', 'sqrt', 'log2', None]
max_depth_range = [None, 50, 100]
min_samples_split_range = [2, 5, 0.01]
min_samples_leaf_range = [1, 5, 0.01]
min_weight_fraction_leaf_range = [0, 0.01, 0.1]
max_leaf_nodes_range = [None, 10, 25]
min_impurity_decrease_range = [0, 0.1, 1]
bootstrap_range = [True, False]
n_jobs_range = [-1]

cross_validator = StratifiedShuffleSplit(n_splits=3,test_size=0.2)

param_grid = dict(n_estimators=n_estimators_range, criterion=criterion_range,
                  max_features=max_features_range, max_depth=max_depth_range,
                  min_samples_split=min_samples_split_range,
                  min_samples_leaf=min_samples_leaf_range,
                  min_weight_fraction_leaf=min_weight_fraction_leaf_range,
                  max_leaf_nodes=max_leaf_nodes_range,
                  min_impurity_decrease=min_impurity_decrease_range,
                  bootstrap=bootstrap_range, n_jobs=n_jobs_range)

grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid,
                    cv=cross_validator, verbose=10, n_jobs=-1)

metadata_args={'metadata': constants.OMNILOG_METADATA,
               'fasta_header': 'Strain',
               'label_header': 'Host',
               'train_header': None,
               'one_vs_all': 'Human',
               'remove': 'Unknown'}
data = get_omnilog_data(metadata_args)
X = np.vstack((data[0],data[2]))
Y = data[1] + data[3]

grid.fit(X, Y)

with open('results.txt', 'w') as f:
    f.write(json.dumps(grid.best_params_, sort_keys=True, indent=4,
                       separators=(',', ': ')))
