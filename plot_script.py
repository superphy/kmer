import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys

def plot(data1, data2, label1, label2, title, max_x):
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

    if len(labels) < 40:
        height = 15
        fig_width = 25
    else:
        height = 20
        fig_width = 30

    total_bars = len(labels)
    spacing = np.linspace(0,height,total_bars+1.0)[1:]
    width = height/(3.0*total_bars)
    data1_y = [x+(width/2) for x in spacing]
    data2_y = [x-(width/2) for x in spacing]

    plt.figure(figsize=(fig_width,height))
    plt.title(title, fontsize=24)
    plt.barh(data1_y, data1_vals, width, zorder=3, label=label1)
    for i, v in enumerate(data1_vals):
        if v > 0:
            if v < 0.01:
                label = '%.4f'%v
            else:
                label = '%.2f'%v
            plt.text(v+(max_x/1000.0), data1_y[i]+0.01, label, va='center', fontweight='bold')
    plt.barh(data2_y, data2_vals, width, zorder=3, label=label2)
    for i, v in enumerate(data2_vals):
        if v > 0:
            if v < 0.01:
                label = '%.4f'%v
            else:
                label = '%.2f'%v
            plt.text(v+(max_x/1000.0), data2_y[i]-0.01, label, va='center', fontweight='bold')
    plt.yticks(spacing, labels, fontsize=12)
    plt.xticks(np.arange(0,max_x,max_x/20), fontsize=12)
    plt.xlabel('Score', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    plt.grid(zorder=2, axis='x')
    plt.legend(fontsize=18)
    plt.ylim((0,max(spacing)+(3*width)))
    plt.xlim((0,max_x))
    plt.tight_layout()
    plt.show()

def plot_data(yaml_file, cutoff=15, max_x=100, subtitle='', o_type=None):
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
            title = 'Important Features for Predicting %s O-Type\nRank Aggregation Done by %s' % (o_type, subtitle)
        plot(svm_data, rf_data, 'SVM', 'Random Forest', title, max_x)
    else:
        for key, value in data.items():
            if 'random_forest' in key:
                continue
            o_type = key.replace('support_vector_machine ', '')
            rf_key = 'random_forest ' + o_type
            svm_data = value
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
                title = 'Important Features for Predicting %s O-Type\nRank Aggregation Done by %s' % (o_type, subtitle)
            plot(svm_data, rf_data, 'SVM', 'Random Forest', title, max_x)

def multivote(o_type=None):
    plot_data('aggr_data/10_votes.yaml', cutoff=15, max_x=1.05, subtitle='Summing the Number of Times a Feature was Selected as Being in the Top 10 Most Important Features', o_type=o_type)

def singlevote(o_type=None):
    plot_data('aggr_data/single_vote.yaml', cutoff=15, max_x=1.05, subtitle='Summing the Number of Times a Feature was Selected as Being the Most Important Feature', o_type=o_type)

def lineardecay(o_type=None):
    plot_data('aggr_data/linear_decay.yaml', cutoff=15, max_x=1.05, subtitle='Applying Linear Decay to Each Ranking and Then Summing Values', o_type=o_type)

def exponentialdecay(o_type=None):
    plot_data('aggr_data/exponential_decay.yaml', cutoff=15, max_x=1.05, subtitle='Applying Exponential Decay to Each Ranking and Then Summing Values', o_type=o_type)

def logarithmicdecay(o_type=None):
    plot_data('aggr_data/log2_decay.yaml', cutoff=15, max_x=1.05, subtitle='Applying Logarithmic Decay to Each Ranking and Then Summing Values', o_type=o_type)

def modelresults(o_type=None):
    plot_data('aggr_data/model_values.yaml', cutoff=15, max_x=75, subtitle='Summing Importance Values From Models', o_type=o_type)


if __name__ == "__main__":
    f = sys.argv[1]
    if len(sys.argv) > 2:
        o = sys.argv[2]
    else:
        o = None
    if 'multi' in f:
        multivote(o)
    elif 'single' in f:
        singlevote(o)
    elif 'line' in f:
        lineardecay(o)
    elif 'ex' in f:
        exponentialdecay(o)
    elif 'log' in f:
        logarithmicdecay(o)
    elif 'model' in f:
        modelresults(o)
    else:
        mutlivote(o)
