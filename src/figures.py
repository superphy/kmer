#!/usr/bin/env python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys

def to_ukus(name):
    if name in ['ukus', 'uk&us']:
        return 'uk and us mixed'
    elif name == 'aCrossValidation':
        return 'a cross validation'
    else:
        return name

if __name__ == "__main__":
    path = sys.argv[1]
    num_samples = sum([len(files) for r, d, files in os.walk(path)])

    master_df = pd.DataFrame(data = np.zeros((num_samples, 6),dtype = 'object'),index =[i for i in range(num_samples)], columns = ['acc','model','feats','train','test','attribute'])
    title_string = ''
    for root, dirs, files in os.walk(path):
        for i, file in enumerate(files):
            """
            All files in directory are in the format:
            Host_2700feats_SVMtrainedOnkmer_testedOnaCrossValidation.pkl
            We split on the _ and extract the relevant information to load into dataframe
            """
            file = file.split('.')[0]
            if len(file.split('_')) == 5:
                attribute, num_feats, train1, train2, test = file.split('_')
                train = train1+'&'+train2
            else:
                attribute, num_feats, train, test = file.split('_')
            num_feats = int(num_feats[:-5]) # all but last 5 chars
            model = train[:3] # first 3 chars
            train = train[12:] # all but first 12
            test = test[8:] # all but last 8
            #print(attribute, num_feats, model, train, test)
            acc_data = pd.read_pickle(path+file+'.pkl')

            """
            We cannot use the average accuracy across the 5 folds because there are a
            different number of samples in each fold. So we multiply class accuracy by
            number of samples in that class, sum that for all classes and divide by total # classes
            """
            acc = 0
            total = np.sum(acc_data.values[:,3])
            for row in acc_data.values:
                acc += row[1]*row[3]/total
            for j, stat in enumerate([acc,model,num_feats,train,test,attribute]):
                master_df.values[i,j] = stat
            if(i==0):
                title_string = (("{} predictor trained on {}, tested on {}".format(attribute, train, test)))

    master_df['feats'] = pd.to_numeric(master_df['feats'])
    master_df['acc'] = pd.to_numeric(master_df['acc'])
    #print(master_df)
    #print(master_df.dtypes)

    if master_df.values.shape[0] > 500:
        # all results currently loaded in df
        sub_df = master_df[(master_df['feats'] == 1000) & (master_df['attribute'].isin(['host','Host'])) & (~master_df['train'].isin(['omnilog','kmer'])) & (~master_df['test'].isin(['omnilog','kmer']))]
        sub_df = sub_df.reset_index()
        sub_df['train'] = pd.Series([to_ukus(i) for i in sub_df['train']])
        sub_df['acc'] = pd.Series([i*100 for i in sub_df['acc']])
        sub_df['setup'] = ['trained on '+sub_df['train'][i]+', tested on '+sub_df['test'][i] for i in range(len(sub_df.index))]
        print(sub_df)
        from collections import Counter
        print(Counter(sub_df['setup']))
        bar = sns.catplot(
        x='setup', y ='acc', hue ='model', kind='bar', data=sub_df,
        hue_order = ["XGB", "SVM", "ANN"],
        order = [
        'trained on uk, tested on aCrossValidation',
        'trained on us, tested on aCrossValidation',
        'trained on uk and us mixed, tested on aCrossValidation',
        'trained on uk, tested on us',
        'trained on us, tested on uk'
        ]
        )
        plt.xlabel("Dataset Setup", fontsize=18)
        plt.ylabel("Accuracy", fontsize=18)
        """
        bar_titles = [
        'Trained on UK, tested\non a cross validation',
        '\n\n\nTrained on UK and US, tested\non a cross validation',
        'Trained on UK,\ntested on US',
        '\n\n\nTrained on US, tested\non a cross validation',
        'Trained on US,\ntested on uk'
        ]
        """
        bar_titles = [
        'UK 5-fold\ncross\nvalidation',
        'US 5-fold\ncross\nvalidation',
        'UK and US\nmixed\n5-fold cross\nvalidation',
        'Trained on\nUK, tested\nwith US',
        'Trained on\nUS, tested\nwith UK'
        ]

        bar.set_xticklabels(bar_titles)
        #plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
        #plt.xticks(rotation=90)
        #plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(11, 8)
        plt.show()

    else:
        # results just for a specific subfolder, i.e. uk_host cv
        idk = sns.relplot(x="feats", y="acc", hue="model", kind="line", data=master_df,
        hue_order = ["XGB", "SVM", "ANN"], legend_out=False)

        plt.rcParams["axes.titlesize"] = 8
        plt.title(title_string)
        plt.ylim(0,1)
        plt.savefig('results/figures/'+(title_string.replace(" ",""))+'.png')
