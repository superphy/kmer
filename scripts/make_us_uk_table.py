import yaml
import pandas as pd
import numpy as np

cols = ['Model', 'Data Type', 'Train/Test Split', 'Accuracy']
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
    split = name[-1].replace('.yml', '')
    data = name[-2]
    model = ' '.join(name[:-2]).title()

    output.loc[count] = [model, data, split, acc]
    count += 1

output.to_csv(snakemake.output[0], index=False, sep='|')

