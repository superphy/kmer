import pandas as pd

def main():
    frames = [pd.read_csv(x, sep='|') for x in snakemake.input]
    complete = pd.concat(frames, ignore_index=True)
    complete = complete[complete.Model != '---']
    new = pd.DataFrame(columns=complete.columns)
    new.loc[0] = ['---',]*len(complete.columns)
    output_df = pd.concat([new, complete], ignore_index=True)
    output_df.to_csv(output[0], sep='|', index=False)

if __name__ == '__main__':
    main()

