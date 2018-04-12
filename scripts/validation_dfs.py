import yaml
import pandas as pd
import numpy as np
from kmerprediction import constants

num_rows = snakemake.config['reps'] * len(snakemake.input)
cols = ['Model Type', 'Kmer Filter Method', 'Dataset',
        'K-mer Length', 'Feature Selection Method', 'Accuracy']
output_df = pd.DataFrame(columns=cols, index=np.arange(num_rows))

count = 0
for yf in snakemake.input:
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
        acc = data['output']['results']
        for a in acc:
            output_df.loc[count] = [model, filter_method, dataset, k, selection, a]
            count += 1
output_df.to_csv(snakemake.output[0], index=False, sep=',')
