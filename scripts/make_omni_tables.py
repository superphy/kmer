import yaml
import pandas as pd
import numpy as np

cols = ['Model', 'Data Type', 'Prediction', 'Accuracy']
index = np.arange(len(snakemake.input))
output = pd.DataFrame(columns=cols, index=index)

output.loc[0] = ['---',]*len(cols)

count = 1
for yf in snakemake.input:
    with open(yf, 'r') as f:
        data = yaml.load(f)
    acc = float(data['output']['avg_result'])*100
    acc = "{0:.2f}%".format(acc)
    name = data['name'].split('/')[-1]
    name = name.split('_')
    prediction = name[-1].replace('.yml', '')
    data = name[-2]
    prediction_class = name[-3]
    model = ' '.join(name[:-3]).title()

    if prediction == 'all':
        prediction = 'All ' + prediction_class + 's'
    else:
        prediction = prediction + '/Non-' + prediction

    output.loc[count] = [model, data, prediction, acc]
    count += 1

output.to_csv(snakemake.output[0], index=False, sep='|')

