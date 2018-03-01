import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

datadir = './kmer_data/'
labels = [x for x in os.listdir(datadir) if '.txt' in x]
data_files = [datadir + x for x in labels]
all_data = [pd.read_csv(x, sep=',', index_col=0, dtype='float64',
                        header=None) for x in data_files]
all_data = [(df.index.values, df.values) for df in all_data]

labels = [x.replace('l', '').replace('.txt', '') for x in labels]

all_data = zip(all_data, labels)
all_data = sorted(all_data, key = lambda x: int(x[1]), reverse=False)
all_data, labels = zip(*all_data)

labels = ['Cutoff: ' + x for x in labels]
all_data = zip(all_data, labels)
palette = sns.blend_palette(sns.color_palette('deep'), len(labels))
sns.set(palette=palette, context='paper')

plt.figure(1, figsize=(20,12.5))

for data, label in all_data:
    x, y = data
    y = y*100
    plt.plot(x, y, label=label, linewidth=4.5)

x_vals = np.arange(3, 32, 1)
y_vals = np.arange(47.5, 90.0, 2.5)

plt.xticks(x_vals, fontsize=15)
plt.yticks(y_vals, fontsize=15)

plt.xlim(min(x_vals)-0.2, max(x_vals)+0.2)
plt.ylim(min(y_vals)-0.5, max(y_vals)+0.5)

plt.xlabel("K-mer Length", fontsize=20)
plt.ylabel("Accuracy (% Correct)", fontsize=20)

plt.legend(fontsize=18, loc='upper right', ncol=3)

plt.title('E. coli Host Prediction Accuracy Using an SVM and Different K-mer Lengths and Cutoffs', fontsize=28)

plt.tight_layout()

plt.savefig('./Figures/KmerLength.pdf')
