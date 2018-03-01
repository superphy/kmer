import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(snakemake.input[0])

col_order = np.unique(df['Model Type'].values)
hue_order = np.unique(df['Data Type'].values)

sns.set(font_scale=1.5)

g = sns.factorplot(x=snakemake.wildcards.prediction, y='Accuracy', hue='Data Type',
                  col='Model Type', kind='box', data=df, legend=False,
                  col_order=col_order, hue_order=hue_order, size=12,
                  fliersize=10, linewidth=2,
                  medianprops={'color': 'black', 'linewidth': 3,
                                'solid_capstyle': 'butt'})
g.set_titles("Using a {col_name}")
g.axes[0][0].legend(loc='lower left', title='Data Type')
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Accuracy When Making Binary {} Predictions".format(snakemake.wildcards.prediction))

plt.savefig(snakemake.output[0], dpi=1200)
