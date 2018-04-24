import yaml
import pandas as pd
import numpy as np
from kmerprediction import constants
from validation_naming import convert_filepath

def main():
    num_rows = snakemake.config['reps'] * len(snakemake.input)
    cols = ['Model', 'Datatype', 'Kmer Filter', 'Kmer Length', 'Fragment Size',
            'Dataset', 'Feature Selection', 'Accuracy']
    output_df = pd.DataFrame(columns=cols, index=np.arange(num_rows))

    count = 0
    for yf in snakemake.input:
        info = convert_filepath(yf)
        with open(yf, 'r') as f:
            data = yaml.load(f)
            acc = data['output']['results']
            for a in acc:
                output_df.loc[count] = [info['model'], info['datatype'],
                                        info['filter'], info['k'],
                                        info['fragment'], info['dataset'],
                                        info['selection'], a]
                count += 1
    output_df.to_csv(snakemake.output[0], index=False, sep=',')

if __name__ == "__main__":
    main()
