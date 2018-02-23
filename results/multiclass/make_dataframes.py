import sys
import pandas as pd
import os
import numpy as np
import yaml

sys.path.append('../../')

def convert_model(model):
    if model.upper() == 'NN':
        out = 'Neural Network'
    elif model.upper() == 'RF':
        out = 'Random Forest'
    elif model.upper() == 'SVM':
        out = 'Support Vector Machine'
    else:
        out = model
    return out

def convert_data(data):
    if data[0].upper() == 'K':
        out = 'K-mer Count'
    elif data[0].upper() == 'O':
        out = 'Omnilog AUC'
    else:
        out = data
    return out

def convert_prediction(prediction):
    out = 'All ' + prediction.title() + 's'
    return out

yaml_dir = './yaml_files/'
yaml_files = [yaml_dir + x for x in os.listdir(yaml_dir) if '.yml' in x]

df = pd.DataFrame(columns=['Model Type', 'Data Type', 'Prediction', 'Accuracy'])

count = 0
for yf in yaml_files:
    with open(yf, 'r') as f:
        for data in yaml.load_all(f):
            name = data['name'].split('+')
            model_type = convert_model(name[0])
            data_type = convert_data(name[1])
            prediction = convert_prediction(name[2])
            accuracies = data['output']['results']
            for acc in accuracies:
                df.loc[count] = [model_type, data_type, prediction, acc]
                count += 1

df.to_csv('results.csv', index=False)

