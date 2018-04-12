from kmerprediction.complete_kmer_counter import get_file_counts
import numpy as np
import pandas as pd

def main():
    file_counts = get_file_counts(snakemake.input[0])
    counts, frequency = np.unique(file_counts, return_counts=True)

    output = pd.DataFrame({'Count': counts, 'Frequency': frequency})
    output.to_csv(snakemake.output[0], sep='\t', index=False)

if __name__ == "__main__":
    main()
