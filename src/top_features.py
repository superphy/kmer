
def find_top_feats(imp_arr,sav_path,top_x):
    """
    Takes in a 2-d importance array from model.py and returns the top X features
    """
    top_feats = []
    while(top_x>0):
        # find the highest scoring value
        m = max(imp_arr[1])

        # pull the index of all kmers tieing the top score
        top_indeces = [i for i, j in enumerate(imp_arr[1]) if j == m]

        # make sure we dont take more than top_x total, this can be removed
        # to keep all tying features in the pipeline but beware, if all are zero it will search through a thousand kmers
        if(len(top_indeces) > top_x):
            top_indeces = top_indeces[:top_x]

        top_x -= len(top_indeces)
        for i in top_indeces:
            top_feats.append(imp_arr[0][i])
            imp_arr[1][i] = 0
    return top_feats

if __name__ =="__main__":
    import os, sys
    import numpy as np
    import pandas as pd

    # where we load the features from data/features
    input_arr = sys.argv[1]
    top_feats_list = []
    for array in os.listdir(input_arr):
        values = array.split("_")

        attribute = values[0]

        features= values[1]
        features = features.split("f")
        features = features[0]

        dataset = values[2]
        dataset = dataset.split("On")
        dataset = dataset[1]

        fold = values[4]
        fold = fold.split(".")
        fold = fold[0]

        #print(features)
        if features == "1000" or features  == "100":

            # where we intend on saving the top 5
            output = sys.argv[2]
            filename = output.split("/")[-1]
            save_loc = output[:-len(filename)]

            # how many features to return
            top_x = sys.argv[3]

            if not os.path.exists(os.path.abspath(os.path.curdir)+"/"+save_loc):
                os.mkdir(os.path.abspath(os.path.curdir)+"/"+save_loc)
            path = input_arr + "/" + array
            feat_imps = np.load(path, allow_pickle = True)

            if(isinstance(feat_imps[0,0],bytes)):
                feat_imps = [[i.decode('utf-8') for i in feat_imps[0]],[float(i.decode('utf-8')) for i in feat_imps[1]]]

            top_feats = find_top_feats(feat_imps,output,int(top_x))
            top_feats.append(attribute)
            top_feats.append(features)
            top_feats.append(dataset)
            top_feats.append(fold)
            #print(top_feats)
            top_feats_list.append(top_feats)


    top_feat_dict = {}
    for i in range(len(top_feats_list)):
        for j in range(0, 5, 1):
            top_feat_dict[top_feats_list[i][j]] = 0
    for i in range(len(top_feats_list)):
        for j in range(0, 5, 1):
            top_feat_dict[top_feats_list[i][j]] += 1

    print(sorted(top_feat_dict.items(), key = lambda kv:(kv[1], kv[0])))

    #df = pd.DataFrame(data = top_feats_list, columns = ["feat1", "feat2", "feat3", "feat4", "feat5", "Attribute", "Features", "Dataset", "Fold"])
    #df.to_csv("data/top_features.csv", index = False)
            #print(top_feats)

            #np.save(output, top_feats)
