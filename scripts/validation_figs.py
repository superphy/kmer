import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# TODO: Add information for second legend explaining what the overlay data is
order = ['split', 'mixed', 'reverse_split', 'UK', 'US']

def base_plot(data):
    sns.set(font_scale=1.5)
    x='Dataset'
    y='Accuracy'
    g = sns.factorplot(x=x, y=y, hue='K-mer Length',
                      col='Model Type', row='Kmer Filter Method', kind='box',
                      data=data, legend=False, size=12, fliersize=10,
                      linewidth=2, medianprops={'color': 'black', 'linewidth': 3,
                                                 'solid_capstyle': 'butt'},
                      col_order=['neural_network', 'support_vector_machine', 'random_forest'],
                      order=order)
    return g

def plot_lupolova_comparison(df, overlay, output_file):
    complete_data = df.loc[df['Datatype'] == 'kmer']
    g = base_plot(complete_data)

    base_args = {'medianprops': {'color': 'red', 'linewidth': 3},
                 'order': order, 'x': 'Dataset', 'y': 'Accuracy',
                 'func': sns.boxplot}

    for i in [0, 1]:
        for j in [0, 1, 2]:
            g = g.map(data=overlay, ax=g.axes[i][j], **base_args)

    g.set_titles("{col_name} with {row_name} Data")
    g.axes[0][0].legend(loc='upper left', title='K-mer Length')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Accuracy When Predicting Host Compared to Lupolova Study")

    plt.savefig(output_file, dpi=1200)

def plot_genome_region_comparison(df, output_file):
    complete_data = df.loc[df['Datatype'] == 'kmer']
    g = base_plot(complete_data)

    overlay = df.loc[df['Datatype'] == 'genome region']
    overlay1 = overlay.loc[df['Feature Selection Method'] == 'kbest']
    overlay2 = overlay.loc[df['Feature Selection Method'] == 'kbest197']
    groupby_cols = ['Model Type', 'Dataset']
    overlay1 = overlay1.groupby(groupby_cols).mean().reset_index()
    overlay2 = overlay2.groupby(groupby_cols).mean().reset_index()
    svm1 = overlay1.loc[overlay1['Model Type'] == 'support_vector_machine']
    nn1 = overlay1.loc[overlay1['Model Type'] == 'neural_network']
    rf1 = overlay1.loc[overlay1['Model Type'] == 'random_forest']
    svm2 = overlay2.loc[overlay2['Model Type'] == 'support_vector_machine']
    nn2 = overlay2.loc[overlay2['Model Type'] == 'neural_network']
    rf2 = overlay2.loc[overlay2['Model Type'] == 'random_forest']
    d1 = {0: nn1, 1: svm1, 2: rf1}
    d2 = {0: nn2, 1: svm2, 2: rf2}

    base_args = {'medianprops': {'linewidth': 3},
                'order': order, 'x': 'Dataset', 'y': 'Accuracy', 'func': sns.boxplot}
    for i in [0, 1]:
        for j in [0, 1, 2]:
            g = g.map(data=d1[j], ax=g.axes[i][j], **base_args, color='red')
            g = g.map(data=d2[j], ax=g.axes[i][j], **base_args, color='orange')

    g.set_titles("{col_name} with {row_name} Data")
    g.axes[0][0].legend(loc='upper left', title='K-mer Length')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Accuracy When Predicting Host Compared to Genome Region Accuracy")

    plt.savefig(output_file, dpi=1200)

def plot_both_comparisons(df, first_overlay, output_file):
    complete_data = df.loc[df['Datatype'] == 'kmer']
    g = base_plot(complete_data)

    overlay = df.loc[df['Datatype'] == 'genome region']
    overlay1 = overlay.loc[df['Feature Selection Method'] == 'kbest']
    overlay2 = overlay.loc[df['Feature Selection Method'] == 'kbest197']
    groupby_cols = ['Model Type', 'Dataset']
    overlay1 = overlay1.groupby(groupby_cols).mean().reset_index()
    overlay2 = overlay2.groupby(groupby_cols).mean().reset_index()
    svm1 = overlay1.loc[overlay1['Model Type'] == 'support_vector_machine']
    nn1 = overlay1.loc[overlay1['Model Type'] == 'neural_network']
    rf1 = overlay1.loc[overlay1['Model Type'] == 'random_forest']
    svm2 = overlay2.loc[overlay2['Model Type'] == 'support_vector_machine']
    nn2 = overlay2.loc[overlay2['Model Type'] == 'neural_network']
    rf2 = overlay2.loc[overlay2['Model Type'] == 'random_forest']
    d1 = {0: nn1, 1: svm1, 2: rf1}
    d2 = {0: nn2, 1: svm2, 2: rf2}

    base_args = {'medianprops': {'linewidth': 3, 'color': 'red'},
                'order': order, 'x': 'Dataset', 'y': 'Accuracy', 'func': sns.boxplot}
    for i in [0, 1]:
        for j in [0, 1, 2]:
            base_args['medianprops']['color'] = 'red'
            g = g.map(data=d1[j], ax=g.axes[i][j], **base_args)
            base_args['medianprops']['color'] = 'orange'
            g = g.map(data=d2[j], ax=g.axes[i][j], **base_args)

    base_args['medianprops']['color'] = 'blue'
    for i in [0, 1]:
        for j in [0, 1, 2]:
            g = g.map(data=first_overlay, ax=g.axes[i][j], **base_args, color='blue')


    g.set_titles("{col_name} with {row_name} Data")
    g.axes[0][0].legend(loc='upper left', title='K-mer Length')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("BLUE: Lupolova Accuracies | RED: Genome Regions K=270" +
                   "| ORANGE: Genome Regions K=197")

    plt.savefig(output_file, dpi=1200)


def main():
    df = pd.read_csv(snakemake.input[0])
    lupolova = pd.read_csv(snakemake.input[1], sep='|',
                           converters={'Accuracy': lambda x: float(x)},
                           skiprows=[1])
    plot_lupolova_comparison(df, lupolova, snakemake.output[0])
    plot_genome_region_comparison(df, snakemake.output[1])
    plot_both_comparisons(df, lupolova, snakemake.output[2])

if __name__ == "__main__":
    main()
