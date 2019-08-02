import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys


def add_lupolova(list, train, attribute, num_feats, acc, model, test):
    #print(train, test)
    if train  == "uk" and test == "us.pkl":
        temp_list = [train, attribute, num_feats, 0.82, "Lupolova"]
        list.append(temp_list)
    else:
        temp_list = [train, attribute, num_feats, 0.78, "Lupolova"]
        list.append(temp_list)

    return list

if __name__ == "__main__":
    directory = sys.argv[1]

    temp_list = []

    for path_to_dir in os.listdir(directory):
        list = []
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
            try:
                attribute, num_feats, train, test = filename.split('_')
            except:
                attribute, num_feats, train, train2, test = filename.split('_')

            num_feats = int(num_feats[:-5]) # all but last 5 chars
            model = train[:3] # first 3 chars
            train = train[12:] # all but first 12
            test = test[8:] # all but last 8

            if 'train2' in locals():
                train = train + train2

            temp_list = [train, attribute, num_feats, acc, model]
            list.append(temp_list)

            list = add_lupolova(list, train, attribute, num_feats, acc, model, test)
        #print(list)
        #master_df = pd.DataFrame(data = list, columns = ["Dataset", "Attribute", "Features", "Accuracy", "Model", "Class"])
        master_df = pd.DataFrame(data = list, columns = ["Dataset", "Attribute", "Features", "Accuracy", "Model"])
        #print(master_df)
        idk = sns.relplot(x="Features", y="Accuracy", hue="Model", kind="line", data=master_df, hue_order = ["XGB", "SVM", "ANN", "Lupolova"])
        title_string = path_to_dir

        if not os.path.exists(os.path.abspath(os.path.curdir)+"/figures_uk_us"):
            os.mkdir(os.path.abspath(os.path.curdir)+"/figures_uk_us")

        plt.title(title_string)
        plt.ylim(0,1)
        plt.tight_layout(pad = 5.5)
        plt.savefig('figures_uk_us/'+(title_string.replace(" ",""))+'.png')
        print(path)

        #print(master_df)
