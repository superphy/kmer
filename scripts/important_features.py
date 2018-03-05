import yaml
import pandas as pd
import numpy as np

important_features = []
models = []

for yf in snakemake.input:
    with open(yf, 'r') as f:
        data = yaml.load(f)
    important_features_dicts = data['output']['important_features']

    curr_model = yf.split('/')[-1]
    curr_model = curr_model.split('_')
    curr_model = ' '.join(curr_model[:-3]).title()
    models.append(curr_model)

    feature_scores = {}

    for feature_dict in important_features_dicts:
        ranked_features = sorted(feature_dict, reverse=True,
                                 key=lambda k: feature_dict[k])

        # The score of a feature is determined by the sum of 
        # of 1/(2^index_i) for each ranking divided by the total
        # number of rankings provided. Where index_i is a feature's 
        # position in ranking i.
        # A total score of 1.0 would indicate that a feature was
        # ranked first overall in every ranking.
        for index, value in enumerate(ranked_features):
            score = (1/(2**index))/len(important_features_dicts)
            if value not in feature_scores:
                feature_scores[value] = 0
            feature_scores[value] += score
    important_features.append(feature_scores)

output_df = pd.DataFrame(columns=['Model', 'Feature', 'Score'])
count = 0

dictionary1 = important_features[0]
dictionary2 = important_features[1]
top_ten_1 = sorted(dictionary1, reverse=True, key=lambda k: dictionary1[k])[:10]
top_ten_2 = sorted(dictionary2, reverse=True, key=lambda k: dictionary2[k])[:10]
top_features = np.unique(np.concatenate((top_ten_1, top_ten_2)))
for elem in top_features:
    if elem in dictionary1:
        output_df.loc[count] = [models[0], elem, dictionary1[elem]]
        count += 1
    if elem in dictionary2:
        output_df.loc[count] = [models[1], elem, dictionary2[elem]]
        count += 1
output_df = output_df.sort_values(by=['Score'], ascending=False)
output_df.to_csv(snakemake.output[0], index=False, sep=',')
