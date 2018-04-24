import yaml
import pandas as pd
import numpy as np
import convert_filepaths

cols = ['Model', 'Datatype', 'Kmer Length', 'Kmer Filter',
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
    info = convert_filepaths.omnilog(yf)
    if info['ova'] == 'All':
        output_prediction = 'All ' + info['prediction'] + 's'
    else:
        output_prediction = info['ova'] + '/Non-' + info['ova']

    output.loc[count] = [info['model'], info['datatype'], info['k'],
                         info['filter'], info['selection'], output_prediction, acc]
    count += 1

output.to_csv(snakemake.output[0], index=False, sep='|')

