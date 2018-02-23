import sys
import os
import yaml
import pandas as pd

sys.path.append('../../')

def convert_model(model):
    if model.upper() == 'NN':
        out = 'Neural Network'
    elif model.upper() == 'SVM':
        out = 'Support Vector Machine'
    elif model.upper() == 'RF':
        out = 'Random Forest'
    else:
        out = model
    return out

def convert_data(data):
    if data.upper() == 'K-MER':
        out = 'K-mer Counts'
    elif data.upper() == 'OMNI':
        out = 'Omnilog AUC'
    else:
        out = data
    return out

def convert_prediction(prediction):
    out = "%s/Non-%s" % (prediction, prediction)
    return out

def gather_yaml_files(name):
    files = [x for x in os.listdir('.') if name == x[:len(name)]]
    return files

def split_on_d_type(df):
    kmer = df.loc[df['Data Type'] == 'K-mer Counts']
    omni = df.loc[df['Data Type'] == 'Omnilog AUC']
    return (kmer, omni)

def make_dataframe(yaml_file):
    df = pd.DataFrame(columns=['Model Type', 'Data Type', 'Prediction',
                               'Accuracy'])
    count = 0
    with open(yaml_file, 'r') as f:
        for doc in yaml.load_all(f):
            name = doc['name']
            name = name.split('+')
            m_type = convert_model(name[0])
            d_type = convert_data(name[1])
            prediction = convert_prediction(name[2])
            results = doc['output']['results']
            for r in results:
                df.loc[count] = [m_type, d_type, prediction, r]
                count += 1
    return df

df_names = ['hosts', 'otypes', 'htypes', 'serotypes']
yaml_files = [gather_yaml_files(x) for x in df_names]
data_frames = [pd.concat([make_dataframe(y) for y in x]) for x in yaml_files]
split_df = [split_on_d_type(x) for x in data_frames]

for i, v in enumerate(split_df):
    v[0].to_csv('%s_kmer.csv' % df_names[i], index=False)
    v[1].to_csv('%s_omni.csv' % df_names[i], index=False)
