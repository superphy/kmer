"""
This serves to average the results of the independent
hyperas tests, for internal use by hyperas.smk
["kmer", "omnilog", "uk", "us", "uk_us"]
"""

import numpy as np
import pandas as pd
import os, sys

if __name__ =="__main__":
    feats = sys.argv[1]
    attribute  = sys.argv[2]
    dataset = sys.argv[3]
    testing_set = sys.argv[4]

    OBN_accs = []
    OBO_accs = []

    index = []
    final = []

    # everything is saved in data/{path}{attribute}/hyperas/
    for i in range(1,6):
        split_df = pd.read_pickle("data/"+dataset+"_"+attribute+"/"+str(feats)+"feats_"+str(i)+"_"+testing_set+"test.pkl")
        print(split_df)

        # initialize new dataframe values
        if i==1:
            final = np.zeros((split_df.shape),dtype=float)
            index = split_df.index

        num_samples = np.sum(split_df.values[:,3])

        # find direct accuracy
        running_sum = 0
        for row in split_df.values:
            running_sum+=(row[1]*row[3]/num_samples)

        # append splits 1-D and direct accuracies to be saved
        OBN_accs.append(running_sum)
        OBO_accs.append(split_df.values[0,4])
        print(OBN_accs, OBO_accs)

        # add values to new dataframe
        for i, row in enumerate(split_df.values):
            for j, cell in enumerate(row):
                final[i,j] += cell

    # average out the cells
    for i, row in enumerate(split_df.values):
        for j, cell in enumerate(row):
            if j == 3:
                continue
            final[i,j] /= 5


    final_df = pd.DataFrame(data = final, index = index, columns = ['Precision','Recall', 'F-Score','Supports', '1D Acc'])
    print(final_df)

    if not os.path.exists(os.path.abspath(os.path.curdir)+"/results/"+dataset+ "2" + testing_set + "_"+attribute+"/"):
        os.mkdir(os.path.abspath(os.path.curdir)+"/results/"+dataset + "2" + testing_set + "_"+attribute+"/")

    final_df.to_pickle("results/"+dataset + "2" + testing_set + "_"+attribute+"/"+attribute + "_" +feats + "feats_ANNtrainedOn"+dataset+"_testedOn"+testing_set+".pkl")

    if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/split_accuracies"):
        os.mkdir(os.path.abspath(os.path.curdir)+"/data/split_accuracies")
    # saving the accuracies for each split
    np.save('data/split_accuracies/'+attribute+'_'+str(feats)+'feats_ANNtrainedOn{}_testedOn'+testing_set+'.npy'.format(attribute) ,np.vstack((OBN_accs,OBO_accs)))
