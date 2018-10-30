from kmerprediction.complete_kmer_counter import get_global_counts
import numpy as np
import pandas as pd

def main():
    global_counts = get_global_counts(snakemake.input[0])
    counts, frequency = np.unique(global_counts, return_counts=True)

    output = pd.DataFrame({'Count': counts, 'Frequency': frequency})
    output.to_csv(snakemake.output[0], sep='\t', index=False)

if __name__ == "__main__":
    main()
