import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(font_scale=1.5)

df = pd.read_csv(snakemake.input[0])

g = sns.factorplot(x='Dataset', y='Accuracy', hue='K-mer Length',
                  col='Model Type', kind='box', data=df, legend=False,
                  size=12, fliersize=10, linewidth=2,
                  medianprops={'color': 'black', 'linewidth': 3,
                                'solid_capstyle': 'butt'},
                  col_order=['Neural Network', 'Support Vector Machine', 'Random Forest'],
                  order=['split', 'mixed'])

# TODO: Get this through snakemake rather than hardcoded in
old_data = '/home/rylan/kmer/filtered_results/US_UK/DataFrames/results.csv'
old_df = pd.read_csv(old_data)
old_df = old_df.loc[old_df['Data Type'] == 'kmer']
old_df = old_df.groupby(['Model Type', 'Data Type', 'Dataset']).mean().reset_index()

svm = old_df.loc[old_df['Model Type'] == 'Support Vector Machine']
nn = old_df.loc[old_df['Model Type'] == 'Neural Network']
rf = old_df.loc[old_df['Model Type'] == 'Random Forest']

g = g.map(sns.boxplot, x='Dataset', y='Accuracy', data=nn, ax=g.axes[0][0],
          medianprops={'color': 'red', 'linewidth': 3}, order=['split', 'mixed'])
g = g.map(sns.boxplot, x='Dataset', y='Accuracy', data=svm, ax=g.axes[0][1],
          medianprops={'color': 'red', 'linewidth': 3}, order=['split', 'mixed'])
g = g.map(sns.boxplot, x='Dataset', y='Accuracy', data=rf, ax=g.axes[0][2],
          medianprops={'color': 'red', 'linewidth': 3}, order=['split', 'mixed'])

g.set_titles("Using a {col_name}")
g.axes[0][0].legend(loc='upper left', title='K-mer Length')
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Accuracy When Making Host Predictions")


plt.savefig(snakemake.output[0], dpi=1200)
