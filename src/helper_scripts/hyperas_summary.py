import pandas as pd
import os

accuracy = 0
path_to_dir = 'results/kmer_Host'
for filename in os.listdir(path_to_dir):
    path = os.path.abspath(path_to_dir+'/'+filename)
    with open(path) as file:
        #print(path)
        if "ANN" in path:
            data = pd.read_pickle(path)
            #print(data)
            acc = sum(data['Recall'] * data['Supports']) / sum(data['Supports'])
            print(acc, path)
            if acc >= accuracy:
                accuracy = acc
                highest = path

#print("accuracy : " + str(accuracy), highest)
