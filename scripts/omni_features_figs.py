import yaml
import numpy as np
import pandas as pd
from kmerprediction.utils import parse_metadata, convert_feature_name
from kmerprediction import constants
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns

def get_distribution(feature, ova, label_header):
    args = {'metadata': constants.OMNILOG_METADATA,
            'fasta_header': 'Strain',
            'train_header': None,
            'label_header': label_header,
            'one_vs_all': ova
           }
    x_train, y_train, x_test, y_test = parse_metadata(**args)

    all_data = list(x_train) + list(x_test)
    all_labels = list(y_train) + list(y_test)

    sample_distributions = {k: [] for k in np.unique(all_labels)}

    for index, value in enumerate(all_data):
        sample_distributions[all_labels[index]].append(value)

    plate_number, well_index = convert_feature_name(feature)
    pattern = r'%s\s(.+)%s' % (plate_number, well_index)
    data = pd.read_csv(constants.OMNILOG_DATA, index_col=0)
    row = data.loc[data.index.str.match(pattern)]

    output = {}
    for key in sample_distributions.keys():
        genomes = [x for x in sample_distributions[key] if x in row]
        data = row[genomes].values
        output[key] = data.reshape(data.shape[1])
    return output

def gather_distribution_data(feature_data, ova, label_header):
    cols = ['Feature', 'Distribution', 'Sample Type']
    data = pd.DataFrame(columns=cols)
    feature_names = feature_data['Feature']
    seen_features = []
    count = 0
    for name in feature_names:
        if name not in seen_features:
            seen_features.append(name)
            distributions = get_distribution(name, ova, label_header)
            for key in distributions:
                for elem in distributions[key]:
                    data.loc[count] = [name, elem, key]
                    count += 1
    return data

def plot_bars(data, palette, ova, p):
    ova = ova or 'All'
    ax = sns.barplot(x='Score', y='Feature', hue='Model', data=data, palette=palette)

    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

    ax.set_xlim(0, 1.01)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_xticklabels(np.arange(0, 1.05, 0.05), fontsize=15)

    ax.set_title('Features Important for predicting {ova} {p}'.format(ova=ova, p=p),
                 fontsize=24)
    legend = ax.legend(title='Model', fontsize=15, loc='lower right')
    plt.setp(legend.get_title(), fontsize=15)

    return ax

def plot_distributions(data, palette):
    ax = sns.stripplot(x='Distribution', y='Feature', hue='Sample Type',
                       data=data, alpha=0.35, size=10, palette=palette)
    ax.set_ylabel('')
    ax.set_yticklabels([])

    x_ticks = np.arange(0, 910, 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=15)
    ax.set_xlim(min(x_ticks)-10, max(x_ticks)+10)
    ax.set_xlabel('Omnilog Are Under the Curve', fontsize=18)

    legend = ax.legend(fontsize=15, loc='lower right', ncol=2)
    plt.setp(legend.get_title(), fontsize=15)

    ax.set_title('Sample Distribution', fontsize=24)
    return ax

def main():
    bar_data = pd.read_csv(snakemake.input[0])
    if snakemake.wildcards.ova == 'all':
        ova = False
    else:
        ova = snakemake.wildcards.ova

    if snakemake.wildcards.prediction == 'Otype':
        label_header = 'O type'
    elif snakemake.wildcards.prediction == 'Htype':
        label_header = 'H type'
    elif snakemake.wildcards.prediction == 'Lineage':
        label_header = 'LSPA6'
    else:
        label_header = snakemake.wildcards.prediction

    dist_data = gather_distribution_data(bar_data, ova, label_header)

    palette1 = sns.color_palette('deep')
    palette2 = sns.color_palette('Set1')
    sns.set(context='paper')

    fig = plt.figure(1, figsize=(25, 12.5))

    plt.subplot(1, 2, 1)
    plot_bars(bar_data, palette1, ova, label_header)

    plt.subplot(1, 2, 2)
    plot_distributions(dist_data, palette2)

    plt.tight_layout()

    plt.savefig(snakemake.output[0])

if __name__ == "__main__":
    main()




