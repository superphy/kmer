import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results.csv')
col_order = np.unique(df['Data Type'].values)
hue_order = np.unique(df['Model Type'].values)
sns.set(font_scale=1.5)
g = sns.factorplot(x='Prediction', y='Accuracy', hue='Model Type',
                   col='Data Type', kind='box', data=df, legend_out=False,
                   col_order=col_order, hue_order=hue_order, size=12,
                   fliersize=10, linewidth=2,
                   medianprops={'color': 'black', 'linewidth': 3,
                                'solid_capstyle': 'butt'})
g.set_titles("Using {col_name} Data")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Model Accuracy for Making Multiclass Predictions")
plt.show()

