import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys


if __name__ == "__main__":
    path_to_dir = sys.argv[1]
    list = []
    temp_list = []

    for filename in os.listdir(path_to_dir):
        path = os.path.abspath(path_to_dir+'/'+filename)
        data = pd.read_pickle(path)

        acc = sum(data['Recall'] * data['Supports']) / sum(data['Supports'])

        attribute, num_feats, train, test = filename.split('_')
        num_feats = int(num_feats[:-5]) # all but last 5 chars
        model = train[:3] # first 3 chars
        train = train[12:] # all but first 12
        test = test[8:] # all but last 8

        temp_list = [train, attribute, num_feats, acc, model]
        list.append(temp_list)
        #print(acc, filename)
        #break

    master_df = pd.DataFrame(data = list, columns = ["Dataset", "Attribute", "Features", "Accuracy", "Model"])
    print(master_df)
