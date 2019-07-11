import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys

def average_files(list):
    highest = 0
    lowest  = 100
    average_data = []
    attribute_list = ["kmer-Host", "kmer-Serotype", "kmer-Otype", "kmer-Htype", "omnilog-Host", "omnilog-Serotype", "omnilog-Otype", "omnilog-Htype"]
    for i in range(100 ,3000, 10):
        for attribute in attribute_list:
            for row in list:
                if row[2] == i and row[5] == attribute:
                    acc_list = []
                    acc_list.append(row[3])
                    avg_acc = sum(acc_list) / len(acc_list)

                    if row[2] > highest:
                        highest = row[2]
                    if row[2] < lowest:
                        lowest = row[2]

            Dataset, Attribute = attribute.split('-')
            temp_list = [Dataset, Attribute, i, avg_acc, highest, lowest]
            average_data.append(temp_list)
    return average_data



if __name__ == "__main__":
    directory = sys.argv[1]
    list = []
    temp_list = []

    for path_to_dir in os.listdir(directory): #results_hyperas
        for dir in os.listdir(os.path.abspath(directory + path_to_dir)): # results1 ... results2 ... results3
            for filename in os.listdir(os.path.abspath(directory + path_to_dir + dir)): # kmer-Host ... kmer-Serotype ... omnilog-Host
                path = os.path.abspath(directory + path_to_dir + dir + '/' + filename)
                data = pd.read_pickle(path)
                #print(data, path)
                acc = sum(data['Recall'] * data['Supports']) / sum(data['Supports'])

                attribute, num_feats, train, test = filename.split('_')
                num_feats = int(num_feats[:-5]) # all but last 5 chars
                model = train[:3] # first 3 chars
                train = train[12:] # all but first 12
                test = test[8:] # all but last 8

                temp_list = [train, attribute, num_feats, acc, model, dir]
                list.append(temp_list)
        average_data = average_files(list)

        master_df = pd.DataFrame(data = average_data, columns = ["Dataset", "Attribute", "Features", "Accuracy", "Highest", "Lowest"])

        idk = sns.relplot(x="Features", y="Accuracy", kind="line", data=master_df)
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
