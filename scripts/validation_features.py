import yaml
import pandas as pd
import numpy as np

def add_to_output(yaml_file, df):
    name = yaml_file.split('/')
    selection = name[-2]
    dataset = name[-3]
    filter_method = '_'.join(name[-4].split('_')[1:])
    model = name[-5]
    with open(yaml_file, 'r') as f:
        data = yaml.load(f)
        data = data['output']['important_features']
    feature_scores = {}
    for d in data:
        ranked_features = sorted(d, reverse=True, key=lambda k: d[k])
        for index, value in enumerate(ranked_features):
            score = (1/(2**index))/len(data)
            if value not in feature_scores:
                feature_scores[value] = 0.0
            feature_scores[value] += score
    height = df.shape[0]
    top_features = sorted(feature_scores, reverse=True, key=lambda k: feature_scores[k])
    for index, feature in enumerate(top_features[:50]):
        score = feature_scores[feature]
        curr_out = [feature, model, dataset, filter_method, score]
        df.loc[height + index] = curr_out
    return df

def main():
    cols = ['Kmer', 'Model', 'Dataset', 'Filter', 'Score']
    output = pd.DataFrame(columns=cols)

    for yaml_file in snakemake.input:
        output = add_to_output(yaml_file, output)

    output.to_csv(snakemake.output[0], index=False)

if __name__ == "__main__":
    main()



