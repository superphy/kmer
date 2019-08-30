import numpy as np

arr1 = np.load("annotation/15mer_data/Host_kmer_feature_ranks.npy", allow_pickle = True)

all_feats = np.asarray(arr1)

top_x_feats = 200

top_feats = []
list = []

while(top_x_feats>0):

    # find the highest scoring value
    m = max(all_feats[1])
    top_indeces = [i for i, j in enumerate(all_feats[1]) if j == m]
    # make sure we dont take more than top_x total, this can be removed
    # to keep all tying features in the pipeline but beware, if all are zero it will search through a thousand kmers

    if(len(top_indeces) > top_x_feats):
        top_indeces = top_indeces[:top_x_feats]
    top_x_feats -= len(top_indeces)

    for i in top_indeces:
        print(all_feats[:,i])
        all_feats[1][i] = 0
