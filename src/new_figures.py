import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys


if __name__ == "__main__":
    directory = sys.argv[1]
    list = []
    temp_list = []

for path_to_dir in os.listdir(directory):
    for filename in os.listdir(os.path.abspath(directory + path_to_dir)):
        path = os.path.abspath(directory + path_to_dir+'/'+filename)
        data = pd.read_pickle(path)
        #print(data, path)
        acc = sum(data['Recall'] * data['Supports']) / sum(data['Supports'])

        '''
        class_acc = data['Recall']
        bovine = class_acc[0]
        human = class_acc[1]
        ovine = class_acc[2]
        water = class_acc[3]
        '''

        attribute, num_feats, train, test = filename.split('_')
        num_feats = int(num_feats[:-5]) # all but last 5 chars
        model = train[:3] # first 3 chars
        train = train[12:] # all but first 12
        test = test[8:] # all but last 8

        temp_list = [train, attribute, num_feats, acc, model]
        list.append(temp_list)
        #print(acc, filename)
        #break


    #master_df = pd.DataFrame(data = list, columns = ["Dataset", "Attribute", "Features", "Accuracy", "Model", "Class"])
    master_df = pd.DataFrame(data = list, columns = ["Dataset", "Attribute", "Features", "Accuracy", "Model"])

    idk = sns.relplot(x="Features", y="Accuracy", hue="Model", kind="line", data=master_df, hue_order = ["XGB", "SVM", "ANN"])
    title_string = "{0} predicted using {1} on a Crossvalidation".format(attribute, train)

    if not os.path.exists(os.path.abspath(os.path.curdir)+"/figures"):
        os.mkdir(os.path.abspath(os.path.curdir)+"/figures")

    plt.title(title_string)
    plt.ylim(0,1)
    plt.tight_layout(pad = 5.5)
    plt.savefig('figures/'+(title_string.replace(" ",""))+'.png')
    print(path)
    break

    #print(master_df)
