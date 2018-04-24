import yaml
import pandas as pd
import numpy as np

cols = ['Model', 'Data Type', 'Kmer Length', 'Kmer Filter',
        'Feature Selection', 'Prediction', 'Accuracy']
index = np.arange(len(snakemake.input) + 1)
output = pd.DataFrame(columns=cols, index=index)

output.loc[0] = ['---',]*len(cols)

count = 1
for yf in snakemake.input:
    with open(yf, 'r') as f:
        data = yaml.load(f)
    acc = float(data['output']['avg_result'])*100
    acc = "{0:.2f}%".format(acc)
    name = yf.replace('results/omnilog/yaml/', '')
    name = name.split('/')
    model = name[0]
    if name[1] == 'omnilog':
        datatype = 'omnilog'
        k = None
        f = None
    else:
        kmer_info = name[1].split('_')
        datatype = 'Kmer'
        k = int(kmer_info[0].replace('mer', ''))
        f = kmer_info[1]
    selection = name[2]
    prediction = name[3]
    ova = name[4]
    if ova == 'all':
        ova = 'All ' + prediction + 's'
    else:
        ova = ova + '/Non-' + ova

    output.loc[count] = [model, datatype, k, f, selection, ova, acc]
    count += 1

output.to_csv(snakemake.output[0], index=False, sep='|')

