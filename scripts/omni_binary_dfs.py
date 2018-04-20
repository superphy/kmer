import yaml
import pandas as pd
import numpy as np
from kmerprediction import constants

def main():
    num_rows = snakemake.config['reps'] * len(snakemake.input)
    cols = ['Model Type', 'Data Type', 'Kmer Length', 'Kmer Filter',
            'Feature Selection', snakemake.wildcards.prediction, 'Accuracy']
    output_df = pd.DataFrame(columns=cols, index=np.arange(num_rows))

    count = 0
    for yf in snakemake.input:
        name = yf.replace('results/omnilog/yaml/', '')
        name = name.split('/')
        ova = name[-2]
        prediction = name[-3]
        selection = name[-4]
        if name[-5] == 'omnilog':
            datatype = 'omnilog'
            k = None
            f = None
        else:
            datatype = 'kmer'
            kmer_info = name[-5].split('_')
            f = kmer_info[1]
            k = int(kmer_info[0].replace('mer', ''))
        model = ' '.join(name[:-5]).title()
        with open(yf, 'r') as f:
            data = yaml.load(f)
            acc = data['output']['results']
            for a in acc:
                curr_row = [model, datatype, k, f, selection, ova, a]
                output_df.loc[count] = curr_row
                count += 1
    output_df.to_csv(snakemake.output[0], index=False, sep=',')

if __name__ == "__main__":
    main()
