import pandas as pd
import numpy as np

def acc(input_file, model=None, data=None, prediction=None, extra_col=None, extra_val=None):
    df = pd.read_csv(input_file)

    if model is not None:
        df = df.loc[df['Model'] == model]
    if data is not None:
        df = df.loc[df['Datatype'] == data]
    if prediction is not None:
        df = df.loc[df['Prediction'] == prediction]
    if extra_col is not None and extra_val is not None:
        if not isinstance(extra_col, list):
            df = df.loc[df[extra_col] == extra_val]
        else:
            for index, col in enumerate(extra_col):
                df = df.loc[df[col] == extra_val[index]]

    df = df['Accuracy'].values

    return (np.mean(df), np.std(df))

def main():
    host_file = snakemake.input[0]
    htype_file = snakemake.input[1]
    lineage_file = snakemake.input[2]
    otype_file = snakemake.input[3]
    serotype_file = snakemake.input[4]
    multiclass_file = snakemake.input[5]

    output = {}

    data = acc(serotype_file, data='kmer')
    output['WGSSerotypeAcc'] = data[0]
    output['WGSSerotypeStdAcc'] = data[1]

    data = acc(serotype_file, data='omni')
    output['OmniSerotypeAcc'] = data[0]
    output['OmniSerotypeStdAcc'] = data[1]

    data1 = acc(multiclass_file, data='kmer', prediction='All Serotype')
    data2 = acc(multiclass_file, data='omni', prediction='All Serotype')
    output['DiffMultiWGSMultiOmniSerotypeAcc'] = data1[0] - data2[0]

    data1 = acc(serotype_file, data='kmer')
    data2 = acc(serotype_file, data='omni')
    output['DiffWGSOmniSerotypeAcc'] = data1[0] - data2[0]

    data1 = acc(serotype_file, model='Random Forest')
    data2 = acc(multiclass_file, model='Random Forest', prediction='All Serotype')
    output['DiffBinMultiSerotypeRFAcc'] = data1[0] - data2[0]

    data1 = acc(serotype_file, model='Support Vector Machine')
    data2 = acc(multiclass_file, model='Support Vector Machine',
    prediction='All Serotype')
    output['DiffBinMultiSerotypeSVMAcc'] = data1[0] - data2[0]

    data = acc(multiclass_file, model='Neural Network', data='kmer', prediction='All Serotype')
    output['MultiWGSSerotypeNNAcc'] = data[0]

    data = acc(multiclass_file, model='Neural Network', data='omni', prediction='All Serotype')
    output['MultiOmniSerotypeNNAcc'] = data[0]

    data1 = acc(otype_file, data='kmer')
    data2 = acc(otype_file, data='omni')
    output['DiffWGSOmniOtypeAcc'] = data1[0] - data2[0]

    data1 = acc(htype_file, data='kmer')
    data2 = acc(htype_file, data='omni')
    output['DiffWGSOmniHtypeAcc'] = data1[0] - data2[0]

    data = acc(multiclass_file, model='Neural Network', prediction='All Otype')
    output['MultiOtypeNNAcc'] = data[0]

    data = acc(multiclass_file, model='Neural Network', prediction='All Htype')
    output['MultiHtypeNNAcc'] = data[0]

    data1 = acc(otype_file, model='Random Forest', data='kmer')[0]
    data1 -= acc(multiclass_file, model='Random Forest', data='kmer', prediction='All Otype')[0]
    data2 = acc(htype_file, model='Random Forest', data='omni')[0]
    data2 -= acc(multiclass_file, model='Random Forest', data='omni', prediction='All Otype')[0]
    data3 = acc(otype_file, model='Support Vector Machine', data='kmer')[0]
    data3 -= acc(multiclass_file, model='Support Vector Machine', data='kmer',
    prediction='All Htype')[0]
    data4 = acc(htype_file, model='Support Vector Machine', data='omni')[0]
    data4 -= acc(multiclass_file, model='Support Vector Machine', data='omni',
    prediction='All Htype')[0]
    data_min = min([data1, data2, data3, data4])
    data_max = max([data1, data2, data3, data4])
    output['MinMultiBinOHtypeRFSVMAcc'] = data_min
    output['MaxMultiBinOHtypeRFSVMAcc'] = data_max

    data = acc(host_file, data='kmer')
    output['WGSHostAcc'] = data[0]

    data = acc(host_file, data='omni')
    output['OmniHostAcc'] = data[0]

    data1 = acc(host_file, extra_col='Host', extra_val='Ovine')[0]
    data2 = acc(host_file, extra_col='Host', extra_val='Water')[0]
    data3 = acc(host_file, extra_col='Host', extra_val='Bovine')[0]
    data4 = acc(host_file, extra_col='Host', extra_val='Human')[0]
    output['MaxDiffSpecificHost'] = max(data1, data2) - min(data3, data4)
    output['MinDiffSpecificHost'] = min(data1, data2) - max(data3, data4)

    output = {k: 100*v for k, v in output.items()}

    macros = ""
    for key, value in data.items():
        macros += '\\newcommand{{\\{}}}{{{:.2f}}}\n'.format(key, value)
    with open(snakemake.output[0], 'w') as f:
        f.write(macros)

if __name__ == "__main__":
    main()
