import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(snakemake.input[0])
genome_region = df.loc[df['Datatype'] == 'Genome region']
genome_region['Data'] = genome_region['Datatype']
kmer = df.loc[df['Datatype'] == 'Kmer']
kmer['Data'] = kmer['Kmer Length'].map(int).map(str) + 'mer'
df = pd.concat([genome_region, kmer], ignore_index=True)

col_order = np.unique(df['Model'].values)
hue_order = np.unique(df['Data'].values)

sns.set(font_scale=1.5)

if snakemake.wildcards.prediction == 'Multiclass':
    x = 'Prediction'
else:
    x = snakemake.wildcards.prediction

plt.figure()
g = sns.factorplot(x=x, y='Accuracy', hue='Data', col='Model', kind='box',
                   data=df, legend_out=False, col_order=col_order,
                   hue_order=hue_order, size=12, fliersize=10, linewidth=2,
                   medianprops={'color': 'black', 'linewidth': 3,
                                'solid_capstyle': 'butt'})
g.set_titles("Using a {col_name}")
plt.subplots_adjust(top=0.9)
prediction = snakemake.wildcards.prediction.title()
g.fig.suptitle("Accuracy When Making {} Predictions".format(prediction))

plt.savefig(snakemake.output[0], dpi=1200)
