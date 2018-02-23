import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

predictions = ['hosts', 'serotypes', 'otypes', 'htypes']

for p in predictions:
    df = pd.read_csv('%s.csv' % p)
    col_order = np.unique(df['Data Type'].values)
    hue_order = np.unique(df['Model Type'].values)
    sns.set(font_scale=1.5)
    g = sns.factorplot(x=p.title()[:-1], y='Accuracy', hue='Model Type', col='Data Type',
                       kind='box', data=df, legend=False, col_order=col_order,
                       size=12, fliersize=10, linewidth=2, hue_order=hue_order,
                       medianprops={'color': 'black', 'linewidth': 3, 'solid_capstyle': 'butt'})
    g.set_titles("Using {col_name} Data")
    g.axes[0][0].legend(loc='lower left')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Accuracy for Making Binary %s Predictions" % p.title()[:-1])
plt.show()

