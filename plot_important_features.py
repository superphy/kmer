"""
Functionality to plot important features and their AUC value distributions.
"""
import argparse
import numpy as np
import pandas as pd
from utils import parse_metadata, convert_feature_name
import yaml
import matplotlib.pyplot as plt
import constants


def get_distribution(o_type, feature):
    """
    Gets the Omnilog AUC values for a passed feature and seperates them into a
    group of values positive for the given o-type and group negative for the
    given o-type.

    Args:
        o_type:
        feature:

    Returns:
        tuple
    """
    args = {'metadata': constants.OMNILOG_METADATA,
            'fasta_header': 'Strain',
            'train_header': None,
            'label_header': 'O type',
            'one_vs_all': o_type
            }
    x_train, y_train, x_test, y_test = parse_metadata(**args)
    all_labels = list(y_train) + list(y_test)
    pos_genomes = []
    neg_genomes = []
    for index, value in enumerate(list(x_train)+list(x_test)):
        if all_labels[index] == o_type:
            pos_genomes.append(value)
        else:
            neg_genomes.append(value)
    plate_number, well_index = convert_feature_name(feature)
    pattern = r'%s\s(.+)%s' % (plate_number, well_index)
    data = pd.read_csv(constants.OMNILOG_DATA, index_col=0)
    row = data.loc[data.index.str.match(pattern)]
    pos_genomes = [x for x in pos_genomes if x in row]
    neg_genomes = [x for x in neg_genomes if x in row]
    pos_data = row[pos_genomes].values
    neg_data = row[neg_genomes].values
    return pos_data, neg_data


def plot(data1, data2, label1, label2, title, max_x, o_type):
    """
    Plots data1, data2 as horizontal bar charts with corresponding labels.

    Args:
        data1:
        data2:
        label1:
        label2:
        title:
        max_x:
        o_type;

    Returns:
        None
    """
    for key in data1.keys() + data2.keys():
        if key not in data1:
            data1[key] = 0
        if key not in data2:
            data2[key] = 0

    labels = sorted(data1.keys(), reverse=True)
    labels = sorted(labels, key=lambda x: data2[x])
    labels = sorted(labels, key=lambda x: data1[x])

    data1_vals = [data1[x] for x in labels]
    data2_vals = [data2[x] for x in labels]

    labels = [unicode(x, 'utf-8') for x in labels]

    height = 15
    fig_width = 35

    total_bars = len(labels)
    spacing = np.linspace(0, height, total_bars+1.0)[1:]
    width = height/(3.0*total_bars)
    data1_y = [x+(width/2) for x in spacing]
    data2_y = [x-(width/2) for x in spacing]

    plt.figure(figsize=(fig_width, height))
    plt.subplot(121)
    plt.title(title, fontsize=24)
    plt.barh(data1_y, data1_vals, width, zorder=3, label=label1)
    for i, v in enumerate(data1_vals):
        if v > 0:
            if v < 0.01:
                label = '%.4f' % v
            else:
                label = '%.2f' % v
            plt.text(v+(max_x/1000.0), data1_y[i]+0.01, label, va='center',
                     fontweight='bold')
    plt.barh(data2_y, data2_vals, width, zorder=3, label=label2)
    for i, v in enumerate(data2_vals):
        if v > 0:
            if v < 0.01:
                label = '%.4f' % v
            else:
                label = '%.2f' % v
            plt.text(v+(max_x/1000.0), data2_y[i]-0.01, label, va='center',
                     fontweight='bold')
    plt.yticks(spacing, labels, fontsize=12)
    plt.xticks(np.arange(0, max_x, max_x/10), fontsize=12)
    plt.xlabel('Score', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    plt.grid(zorder=2)
    plt.legend(fontsize=18)
    plt.ylim((0, max(spacing)+(3*width)))
    plt.xlim((0, max_x))
    plt.tight_layout()
    plt.subplot(122)
    for i, value in enumerate(labels):
        pos, neg = get_distribution(o_type, value)
        plt.plot(pos, np.zeros_like(pos)+spacing[i], 'o', color='blue',
                 zorder=4, alpha=0.4)
        plt.plot(neg, np.zeros_like(neg)+spacing[i], 'o', color='red',
                 zorder=3, alpha=0.4)
    plt.plot(np.NaN, np.NaN, 'o', color='blue', label=o_type + ' Sample')
    plt.plot(np.NaN, np.NaN, 'o', color='red', label='Non-%s Sample' % o_type)
    plt.legend(fontsize=18)
    plt.yticks(spacing, [])
    plt.grid(zorder=2, axis='y')
    plt.ylim((0, max(spacing)+(3*width)))
    plt.xlabel('Omnilog Area Under the Curve', fontsize=18)
    plt.title('Feature Distribution by Sample Classification', fontsize=24)
    plt.show()


def plot_data(yaml_file, cutoff=15, max_x=100, subtitle='', o_type=None):
    """
    Wraps plot to plot the data contained in yaml_file.

    Args:
        yaml_file:
        cutoff:
        max_x:
        subtitle:
        o_type:

    Returns:
        None
    """
    with open(yaml_file, 'r') as f:
        data = yaml.load(f)

    if o_type:
        svm_key = 'support_vector_machine ' + o_type
        rf_key = 'random_forest ' + o_type
        svm_data = data[svm_key]
        rf_data = data[rf_key]
        svm_sort = sorted(svm_data.items(), key=lambda x: x[1], reverse=True)
        rf_sort = sorted(rf_data.items(), key=lambda x: x[1], reverse=True)
        if len(svm_sort) > cutoff:
            svm_sort = svm_sort[:cutoff]
        if len(rf_sort) > cutoff:
            rf_sort = rf_sort[:cutoff]
        svm_data = dict(svm_sort)
        rf_data = dict(rf_sort)
        if not subtitle:
            title = 'Important Features for Predicting %s O-Type' % o_type
        else:
            title = ('Important Features for Predicting %s' % o_type +
                     'O-Type\nRank Aggregation Done by %s' % subtitle)
        plot(svm_data, rf_data, 'SVM', 'Random Forest', title, max_x, o_type)
    else:
        for key, value in data.items():
            if 'random_forest' in key:
                continue
            o_type = key.replace('support_vector_machine ', '')
            rf_key = 'random_forest ' + o_type
            svm_data = value
            rf_data = data[rf_key]
            svm_sort = sorted(svm_data.items(), key=lambda x: x[1],
                              reverse=True)
            rf_sort = sorted(rf_data.items(), key=lambda x: x[1], reverse=True)
            if len(svm_sort) > cutoff:
                svm_sort = svm_sort[:cutoff]
            if len(rf_sort) > cutoff:
                rf_sort = rf_sort[:cutoff]
            svm_data = dict(svm_sort)
            rf_data = dict(rf_sort)
            if not subtitle:
                title = 'Important Features for Predicting %s O-Type' % o_type
            else:
                title = ('Important Features for Predicting %s' % o_type +
                         'O-Type\nRank Aggregation Done by %s' % subtitle)
            plot(svm_data, rf_data, 'SVM', 'Random Forest', title, max_x,
                 o_type)


def multivote(o_type=None):
    """
    Wraps plot_data to plot the multivote data.
    """
    plot_data('aggr_data/10_votes.yaml', cutoff=15, max_x=1.05,
              subtitle='10 Equal Votes per Run', o_type=o_type)


def singlevote(o_type=None):
    """
    Wraps plot_data to plot the single vote data..
    """
    plot_data('aggr_data/single_vote.yaml', cutoff=15, max_x=1.05,
              subtitle='Single Vote per Run', o_type=o_type)


def lineardecay(o_type=None):
    """
    Wraps plot_data to plot the linear decay data.
    """
    plot_data('aggr_data/linear_decay.yaml', cutoff=15, max_x=1.05,
              subtitle='Applying Linear Decay to Each Ranking', o_type=o_type)


def exponentialdecay(o_type=None):
    """
    Wraps plot_data to plot the exponential decay data.
    """
    plot_data('aggr_data/exponential_decay.yaml', cutoff=15, max_x=1.05,
              subtitle='Applying Exponential Decay to Each Ranking',
              o_type=o_type)


def logarithmicdecay(o_type=None):
    """
    Wraps plot_data to plot the logarithmic decay data.
    """
    plot_data('aggr_data/log2_decay.yaml', cutoff=15, max_x=1.05,
              subtitle='Applying Logarithmic Decay to Each Ranking',
              o_type=o_type)


def modelresults(o_type=None):
    """
    Wraps plot_data to plot the model value data.
    """
    plot_data('aggr_data/model_values.yaml', cutoff=15, max_x=75,
              subtitle='Summing Importance Values From Models', o_type=o_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--otype", default=None)
    parser.add_argument("-r", "--rankaggr", default=None)
    cl_args = parser.parse_args()
    otype = cl_args.otype
    rankaggr = cl_args.rankaggr
    if not otype and not rankaggr:
        multivote()
        singlevote()
        lineardecay()
        exponentialdecay()
        logarithmicdecay()
        modelresults()
    elif otype and not rankaggr:
        multivote(otype)
        singlevote(otype)
        lineardecay(otype)
        exponentialdecay(otype)
        logarithmicdecay(otype)
        modelresults(otype)
    elif rankaggr and not otype:
        if 'multi' in rankaggr:
            multivote()
        elif 'single' in rankaggr:
            singlevote()
        elif 'line' in rankaggr:
            lineardecay()
        elif 'ex' in rankaggr:
            exponentialdecay()
        elif 'log' in rankaggr:
            logarithmicdecay()
        elif 'model' in rankaggr:
            modelresults()
        else:
            print 'Oops'
    else:
        if 'multi' in rankaggr:
            multivote(otype)
        elif 'single' in rankaggr:
            singlevote(otype)
        elif 'line' in rankaggr:
            lineardecay(otype)
        elif 'ex' in rankaggr:
            exponentialdecay(otype)
        elif 'log' in rankaggr:
            logarithmicdecay(otype)
        elif 'model' in rankaggr:
            modelresults(otype)
        else:
            print 'Oops'
