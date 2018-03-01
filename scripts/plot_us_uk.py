import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(font_scale=1.5)

g = sns.factorplot(x='Dataset', y='Accuracy', hue='Data Type',
                  col='Model Type', kind='box', data=df, legend=False,
                  col_order=col_order, hue_order=hue_order, size=12,
                  fliersize=10, linewidth=2,
                  medianprops={'color': 'black', 'linewidth': 3,
                                'solid_capstyle': 'butt'})
g.set_titles("Using a {col_name}")
g.axes[0][0].legend(loc='lower left', title='Data Type')
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Accuracy When Making Host Predictions")

plt.savefig(snakemake.output[0], dpi=1200)
