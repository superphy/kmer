import yaml
import pandas as pd
import numpy as np

cols = ['Model', 'Kmer Filter Method', 'Dataset', 'k',
        'Feature Selection', 'Accuracy']
index = np.arange(len(snakemake.input))
output = pd.DataFrame(columns=cols, index=index)

output.loc[0] = ['---',]*len(cols)

for index, yf in enumerate(snakemake.input):
    name = yf.replace('results/validation/yaml/', '')
    name = name.split('/')
    model = name[0]
    datatype = name[1].split('_')
    k = int(datatype[0].replace('mer', ''))
    filter_method = '_'.join(datatype[1:])
    dataset = name[2]
    selection = name[3]
    with open(yf, 'r') as f:
        data = yaml.load(f)
    acc = float(data['output']['avg_result'])*100
    acc = "{0:.2f}%".format(acc)

    output.loc[index + 1] = [model, filter_method, dataset, k, selection, acc]

output.to_csv(snakemake.output[0], index=False, sep='|')

