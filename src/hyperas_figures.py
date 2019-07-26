import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys


if __name__ == "__main__":
    directory = sys.argv[1]
    acc_list = []
    avg_list = []
    average_data = []
    final_data = []

    for path_to_dir in os.listdir(directory): #results_hyperas
        for dir in os.listdir(os.path.abspath(directory + path_to_dir)): # results1 ... results2 ... results3
            for filename in os.listdir(os.path.abspath(directory + path_to_dir + '/' + dir)): # kmer-Host ... kmer-Serotype ... omnilog-Host
                path = os.path.abspath(directory + path_to_dir + '/' + dir + '/' + filename)
                data = pd.read_pickle(path)
                acc = sum(data['Recall'] * data['Supports']) / sum(data['Supports'])

                attribute, num_feats, train, test = filename.split('_')
                num_feats = int(num_feats[:-5]) # all but last 5 chars
                model = train[:3] # first 3 chars
                train = train[12:] # all but first 12
                test = test[8:] # all but last 8
                temp_list = [path_to_dir, dir, acc, num_feats]
                acc_list.append(temp_list)

    length = 0
    for x in acc_list:
        length = length + 1

    #for attribute in ["kmer_Host", "kmer_Serotype", "kmer_Otype", "kmer_Htype", "omnilog_Host", "omnilog_Serotype", "omnilog_Otype", "omnilog_Htype"]:
    for i in range(length - 1):
        for j in range(length - 1):
            if acc_list[i][1] == acc_list[j][1] and acc_list[i][3] == acc_list[j][3]:
                avg_list.append(acc_list[j][2])
        avg = sum(avg_list) / len(avg_list)
        temp_list = [acc_list[i][1], acc_list[i][3], avg]
        average_data.append(temp_list)

    for i in range(0, 188, 1):
        final_data.append(average_data[i])
    for list in final_data:
        print(list)

    '''
    for list in acc_list:
        print(list)
    #for attribute in ["kmer_Host", "kmer_Serotype", "kmer_Otype", "kmer_Htype", "omnilog_Host", "omnilog_Serotype", "omnilog_Otype", "omnilog_Htype"]:
    for attribute in ["omnilog_Serotype"]:
        attribute_list = []
        for i in range(length - 1):
            if acc_list[i][1] == attribute:
                attribute_list.append(acc_list[i])
                test, train = attribute.split('_')
                #print(test, train)

        df = pd.DataFrame(data = attribute_list, columns = ["Run", "Data", "Accuracy", "Features"])
    '''

    master_df = pd.DataFrame(data = average_data, columns = ["Attribute", "Features", "Accuracy"])

    idk = sns.relplot(x="Features", y="Accuracy", hue = "Attribute", kind="line", data=master_df, hue_order = ["kmer_Host", "kmer_Serotype", "kmer_Otype", "kmer_Htype", "omnilog_Host", "omnilog_Serotype", "omnilog_Otype", "omnilog_Htype"])

    title_string = "{0} predicted using {1} on a Crossvalidation".format(train, test)

        #kmer = sns.relplot(x="Features", y="Accuracy", hue = "Run", kind="line", data=df, hue_order = ["results1", "results2", "results3", "results4", "results5"])

    if not os.path.exists(os.path.abspath(os.path.curdir)+"/figures"):
        os.mkdir(os.path.abspath(os.path.curdir)+"/figures")

    plt.title(title_string)
    plt.ylim(0,1)
    #plt.tight_layout(pad = 5.5)
    plt.savefig('figures/'+train+'_'+test+'.png')
    plt.clf()
    print(path)
