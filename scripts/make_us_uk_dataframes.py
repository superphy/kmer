import yaml
import pandas as pd
import numpy as np
from kmerprediction import constants

num_rows = constants.DEFAULT_REPITITIONS * len(snakemake.input)
output_df = pd.DataFrame(columns=['Model Type', 'Data Type', 'Dataset', 'Accuracy'],
                         index=np.arange(num_rows))
count = 0
for yf in snakemake.input:
    name = yf.split('/')[-1]
    name = name.split('_')
    dataset = name[-1].replace('.yml', '')
    data_type = name[-2]
    model_type = ' '.join(name[:-3]).title()
    with open(yf, 'r') as f:
        data = yaml.load(f)
        acc = data['output']['results']
        for a in acc:
            output_df.loc[count] = [model_type, data_type, dataset, a]
            count += 1
output_df.to_csv(snakemake.output[0], index=False, sep=',')
