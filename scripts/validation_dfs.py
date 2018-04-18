import yaml
import pandas as pd
import numpy as np
from kmerprediction import constants

def main():
    num_rows = snakemake.config['reps'] * len(snakemake.input)
    cols = ['Model Type', 'Datatype', 'Kmer Filter Method', 'Dataset',
            'K-mer Length', 'Fragment Size', 'Feature Selection Method', 'Accuracy']
    output_df = pd.DataFrame(columns=cols, index=np.arange(num_rows))

    count = 0
    for yf in snakemake.input:
        name = yf.replace('results/validation/yaml/', '')
        name = name.split('/')
        model = name[0]
        dataset = name[2]
        selection = name[3]
        if 'mer' in name[1] and 'genome' not in name[1]:
            datatype = name[1].split('_')
            k = int(datatype[0].replace('mer', ''))
            filter_method = '_'.join(datatype[1:])
            fragment = None
            datatype = 'kmer'
        else:
            k = None
            filter_method = None
            fragment = int(name[1].replace('genome', ''))
            datatype = 'genome region'
        with open(yf, 'r') as f:
            data = yaml.load(f)
            acc = data['output']['results']
            for a in acc:
                output_df.loc[count] = [model, datatype, filter_method, dataset,
                                        k, fragment, selection, a]
                count += 1
    output_df.to_csv(snakemake.output[0], index=False, sep=',')

if __name__ == "__main__":
    main()
