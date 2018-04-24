import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np

order = ['Split', 'Mixed', 'Reverse split', 'UK', 'US']
col_order = ['Neural network', 'Support vector machine', 'Random forest']

def base_plot(data):
    sns.set(font_scale=1.5)
    x='Dataset'
    y='Accuracy'
    g = sns.factorplot(x=x, y=y, hue='Kmer Length',
                      col='Model', row='Kmer Filter', kind='box',
                      data=data, legend=False, size=12, fliersize=10,
                      linewidth=2, col_order=col_order, order=order,
                      medianprops={'color': 'black', 'linewidth': 3,
                                   'solid_capstyle': 'butt'})
    g.set_titles("{col_name} with {row_name} Data")
    g.axes[0][0].legend(loc='upper left', title='Kmer Length')
    return g

def add_constant_overlay(g, overlay, color):
    args = {'medianprops': {'color': color, 'linewidth': 3},
            'order': order, 'x': 'Dataset', 'y': 'Accuracy',
            'func': sns.boxplot}
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            g = g.map(data=overlay, ax=g.axes[i][j], **args)
    return g

def add_variable_overlay(g, overlays, color, axis):
    assert len(overlays) == g.shape[axis]
    args = {'medianprops': {'color': color, 'linewidth': 3},
            'order': order, 'x': 'Dataset', 'y': 'Accuracy',
            'func': sns.boxplot}
    if axis == 0:
        for j in range(g.shape[1]):
            for i in range(g.shape[0]):
                g = g.map(data=overlays[i], ax=g.axes[i][j], **args)
    else:
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                g = g.map(data=overlays[j], ax=g.axes[i][j], **args)
    return g

def plot_lupolova(outputfile):
    data = pd.read_csv(snakemake.input[0])
    lupolova = pd.read_csv(snakemake.input[1], sep='|', skiprows=[1],
                           converters={'Accuracy': lambda x: float(x)})
    g = base_plot(data)
    color = 'red'
    g = add_constant_overlay(g, lupolova, color)
    patch = Patch(color=color, label='Lupolova Accuracies')
    g.axes[0][0].legend(loc='upper right', handles=[patch])
    plt.savefig(outputfile, dpi=1200)

def get_overlays(data, selection):
    base = data.loc[data['Datatype'] == 'Genome region']
    base = data.loc[data['Feature Selection'] == selection]
    base = base.groupby(by=['Model', 'Dataset']).mean().reset_index()
    overlays = []
    for model in col_order:
        overlays.append(base.loc[base['Model'] == model])
    return overlays

def plot_genome_region(outputfile):
    data = pd.read_csv(snakemake.input[0])
    g = base_plot(data)
    colors = ['red', 'orange']
    patches = []
    for i, selection in enumerate(np.unique(base['Feature Selection'].values)):
        overlays = get_overlays(data, selection)
        g = g.add_variable_overlay(g, overlays, colors[i], 1)
        patches.append(Patch(color=colors[i], label=selection))
    g.axes[0][0].legend(loc='upper right', handles=patches)
    plt.savefig(outputfile, dpi=1200)

def plot_both(outputfile):
    data = pd.read_csv(snakemake.input[0])
    lupolova = pd.read_csv(snakemake.input[1], sep='|', skiprows=[1],
                           converters={'Accuracy': lambda x: float(x)})
    g = base_plot(data)
    g = g.add_constant_overlay(g, lupolova, 'blue')
    patches = [Patch(color='blue', label='Lupolova Accuracies')]
    colors = ['red', 'orange']
    for i, selection in enumerate(np.unique(base['Feature Selection'].values)):
        overlays = get_overlays(data, selection)
        g = g.add_variable_overlay(g, overlays, colors[i], 1)
        patches.append(Patch(color=colors[i], label=selection))
    g.axes[0][0].legend(loc='upper left', handles=patches)
    plt.savefig(outputfile, dpi=1200)

def main():
    plot_lupolova(snakemake.output[0])
    plot_genome_regione(snakemake.output[1])
    plot_both(snakemake.output[2])

if __name__ == "__main__":
    main()
