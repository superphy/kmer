#!/usr/bin/env python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys

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
    print(master_df)
    print(master_df.dtypes)
    idk = sns.relplot(x="feats", y="acc", hue="model", kind="line", data=master_df, hue_order = ["XGB", "SVM", "ANN"])

    plt.rcParams["axes.titlesize"] = 8
    plt.title(title_string)
    plt.ylim(0,1)
    plt.savefig('figures/'+(title_string.replace(" ",""))+'.png')
