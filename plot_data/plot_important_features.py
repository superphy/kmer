import sys
sys.path.append('../')
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils import parse_metadata, convert_feature_name
import constants

def get_distribution(o_type, feature):
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

def get_x_best(input_dict, x):
    sorted_items = sorted(input_dict.items(), key=lambda x: x[1],
                         reverse=True)
    top_items = sorted_items[:x]
    return dict(top_items)

def fill(best, total, other_best):
    for key in other_best.keys():
        if key not in best:
            if key in total:
                best[key] = total[key]
            else:
                best[key] = 0.0
    return best

def gather_feature_data(yaml_file, otype):
    with open(yaml_file, 'r') as f:
        data = yaml.load(f)

    rf_data = data['random_forest %s' % otype]
    best_rf = get_x_best(rf_data, 10)
    svm_data = data['support_vector_machine %s' % otype]
    best_svm = get_x_best(svm_data, 10)

    best_rf = fill(best_rf, rf_data, best_svm)
    best_svm = fill(best_svm, svm_data, best_rf)

    index = np.arange(len(best_rf.keys()) + len(best_svm.keys()))
    cols = ['Feature', 'Score', 'Model']
    data = pd.DataFrame(index=index, columns=cols)

    count = 0
    for key, value in best_rf.items():
        data.loc[count] = [unicode(key, 'utf-8'), value, 'Random Forest']
        count += 1

    for key, value in best_svm.items():
        data.loc[count] = [unicode(key, 'utf-8'), value,
                           'Support Vector Machine']
        count += 1

    data = data.sort_values(['Score'], ascending=False)

    return data

def gather_distribution_data(feature_data, otype):
    cols = ['Feature', 'Distribution', 'Sample Type']
    data = pd.DataFrame(columns=cols)

    feature_names = feature_data['Feature']
    seen_features = []
    count = 0
    for name in feature_names:
        if name not in seen_features:
            seen_features.append(name)
            pos, neg = get_distribution(otype, name)
            pos = pos.reshape(pos.shape[1])
            neg = neg.reshape(neg.shape[1])
            for elem in pos:
                data.loc[count] = [name, elem, 'Positive']
                count += 1
            for elem in neg:
                data.loc[count] = [name, elem, 'Negative']
                count += 1
    return data

def plot_bars(data):
    ax = sns.barplot(x='Score', y='Feature', hue='Model', data=data)

    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

    ax.set_xlim(0, 1.01)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_xticklabels(np.arange(0,1.05,0.05), fontsize=15)

    ax.set_title('Features Important for Predicting O157 O-type',
                 fontsize=24)

    legend = ax.legend(title='Model', fontsize=15, loc='lower right')
    plt.setp(legend.get_title(), fontsize=15)

    return ax

def plot_distributions(data):
    palette = sns.diverging_palette(240, 10, n=2)
    ax = sns.stripplot(x='Distribution', y='Feature', hue='Sample Type',
                       data=data, alpha=0.35, size=10, palette=palette)
    ax.set_ylabel('')
    ax.set_yticklabels([])

    x_ticks = np.arange(0, 910, 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=15)
    ax.set_xlim(min(x_ticks)-10, max(x_ticks)+10)
    ax.set_xlabel('Omnilog Area Under the Curve', fontsize=18)

    legend = ax.legend(fontsize=15, loc='lower right', ncol=2)
    plt.setp(legend.get_title(), fontsize=15)

    ax.set_title('Sample Distribution', fontsize=24)
    return ax

yaml_file = 'important_feature_data/single_vote.yaml' # Set the rank aggregation method
otype = 'O157' # Set the Otype to plot
bar_data = gather_feature_data(yaml_file, otype)
dist_data = gather_distribution_data(bar_data, otype)

palette = sns.color_palette('deep')
sns.set(palette=palette, context='paper')

fig = plt.figure(1, figsize=(25,12.5))

plt.subplot(1,2,1)
plot_bars(bar_data)

plt.subplot(1,2,2)
plot_distributions(dist_data)

plt.tight_layout()

plt.show()
