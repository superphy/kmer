#!/usr/bin/env python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys

num_samples = sum([len(files) for r, d, files in os.walk("results/")])

master_df = pd.DataFrame(data = np.zeros((num_samples, 6),dtype = 'object'),index =[i for i in range(num_samples)], columns = ['acc','model','feats','train','test','attribute'])

for root, dirs, files in os.walk("results/"):
    for i, file in enumerate(files):
        file = file.split('.')[0]
        attribute, num_feats, train, test = file.split('_')
        num_feats = int(num_feats[:-5])
        model = train[:3]
        train = train[12:]
        test = test[8:]
        #print(attribute, num_feats, model, train, test)
        acc_data = pd.read_pickle("results/"+file+'.pkl')
        acc = 0
        total = np.sum(acc_data.values[:,3])
        for row in acc_data.values:
            acc += row[1]*row[3]/total
        for j, stat in enumerate([acc,model,num_feats,train,test,attribute]):
            master_df.values[i,j] = stat

master_df['feats'] = pd.to_numeric(master_df['feats'])
master_df['acc'] = pd.to_numeric(master_df['acc'])
print(master_df)
print(master_df.dtypes)
#fmri = sns.load_dataset("fmri")
#print(fmri)
#idk = sns.relplot(x="timepoint", y="signal", hue="region", style="event", kind="line", data=fmri)
idk = sns.relplot(x="feats", y="acc", hue="model", kind="line", data=master_df)
#idk = sns.relplot(x="feats", y="acc", kind="line", data=master_df)
plt.show()
