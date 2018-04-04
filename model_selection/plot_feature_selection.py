import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def prepare_dir(directory):
    files = [directory + x for x in os.listdir(directory) if '.txt' in x]
    files = [x for x in files if 'kmer' in x]
    files = [x for x in files if 'svm' in x]
    data = [pd.read_csv(x, sep=',', index_col=0, header=None,
                        dtype='float64') for x in files]
    data = [df.apply(lambda x: np.mean(x), axis=1) for df in data]
    data = pd.concat(data, axis=1)
    data = data.apply(lambda x: np.mean(x), axis=1)
    return (data.index.values, data.values)

datadir1 = './feature_selection_data/k_best_f/'
datadir2 = './feature_selection_data/k_best_chi2/'
datadir3 = './feature_selection_data/RFE/'
dirs = [datadir1, datadir2, datadir3]

all_data = [prepare_dir(x) for x in dirs]

labels = ['SelectKBest by F-Test', 'SelectKBest by Chi-Squared Test',
          'Recursive Feature Elimination by SVM']

all_data = zip(all_data, labels)

palette = sns.blend_palette(sns.color_palette('deep'), len(labels))
sns.set(palette=palette, context='paper')

plt.figure(1, figsize=(20, 12.5))

for data, label in all_data:
    x, y = data
    y = y*100
    plt.plot(x, y, label=label, linewidth=4.5)

x_vals = np.arange(0, 1001, 50)
y_vals = np.arange(50.0, 90.0, 2.5)

plt.xticks(x_vals, fontsize=15)
plt.yticks(y_vals, fontsize=15)

plt.xlim(min(x_vals)-10, max(x_vals)+10)
plt.ylim(min(y_vals)-0.5, max(y_vals)+0.5)

plt.xlabel("Features Remaining After Selection", fontsize=20)
plt.ylabel("Accuracy (% Correct)", fontsize=20)

plt.legend(fontsize=18, loc='upper right')

plt.title('E. coli Host Prediction Accuracy Using an SVM and Different Feature Selection Methods', fontsize=28)

plt.tight_layout()

plt.savefig('./Figures/FeatureSelection.pdf')
