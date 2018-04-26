import yaml
import pandas as pd
import numpy as np

def main():
    data = pd.read_csv(snakemake.input[0])
    cols = ['Model', 'Datatype', 'Kmer Filter', 'Kmer Length', 'Fragment Size',
            'Dataset', 'Feature Selection']

    data = data.groupby(by=cols).mean().reset_index()

    header = pd.DataFrame(columns=cols  + ['Mean Accuracy'])
    header.loc[0] = ['---',]*(len(cols) + 1)
    frames = [header, data]
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(snakemake.output[0], index=False, sep='|')

if __name__ == "__main__":
    main()
