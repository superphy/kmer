import pandas as pd
import numpy as np

def acc(input_file, model=None, data=None, prediction=None, extra_col=None, extra_val=None,
        invalid_col=None, invalid_val=None):
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
    if invalid_col is not None and invalid_val is not None:
        for index, col in enumerate(invalid_col):
            df = df.loc[df[col] != invalid_val[index]]

    df = df['Accuracy'].values

    return (np.mean(df), np.std(df))

def main():
    host_file = snakemake.input[0]
    htype_file = snakemake.input[1]
    lineage_file = snakemake.input[2]
    otype_file = snakemake.input[3]
    serotype_file = snakemake.input[4]

    output = {}

    # Average/std accuracy for kmer serotype predictions for all models
    data = acc(serotype_file, data='Kmer', extra_col=['Kmer Length'], extra_val=[7])
    output['WGSSerotypeAcc'] = data[0]
    output['WGSSerotypeStdAcc'] = data[1]


    # Average/std accuracy for omnilog serotype predictions for all models
    data = acc(serotype_file, data='Omnilog')
    output['OmniSerotypeAcc'] = data[0]
    output['OmniSerotypeStdAcc'] = data[1]


    # Difference between average omnilog serotype accuracy and kmer serotype accuracy 
    # For multiclass predictions
    data1 = acc(serotype_file, data='Kmer', extra_col=['Kmer Length', 'Serotype'],
                extra_val=[7, 'all'])
    data2 = acc(serotype_file, data='Omnilog', extra_col=['Serotype'],
                extra_val=['all'])
    output['DiffMultiWGSMultiOmniSerotypeAcc'] = data1[0] - data2[0]


    # Difference between average omnilog serotype accuracy and kmer serotype accuracy 
    # For binary predictions
    data1 = acc(serotype_file, data='Kmer', extra_col=['Kmer Length'], extra_val=[7],
               invalid_col=['Serotype'], invalid_val=['all'])
    data2 = acc(serotype_file, data='Omnilog', invalid_col=['Serotype'], invalid_val=['all'])
    output['DiffWGSOmniSerotypeAcc'] = data1[0] - data2[0]


    # Difference between binary and multiclass serotype predictions using a random forest
    data1 = acc(serotype_file, model='Random forest', extra_col=['Kmer Length'], extra_val=[7],
                invalid_col=['Serotype'], invalid_val=['all'])
    data2 = acc(serotype_file, model='Random forest', extra_col=['Kmer Length', 'Serotype'],
                extra_val=[7, 'all'])
    output['DiffBinMultiSerotypeRFAcc'] = data1[0] - data2[0]


    # Difference between binary and multiclass serotype predictions using a support vector machine
    data1 = acc(serotype_file, model='Support vector machine', extra_col=['Kmer Length'],
                extra_val=[7], invalid_col=['Serotype'], invalid_val=['all'])
    data2 = acc(serotype_file, model='Support vector machine', extra_col=['Kmer Length', 'Serotype'],
                extra_val=[7, 'all'])
    output['DiffBinMultiSerotypeSVMAcc'] = data1[0] - data2[0]


    # mean Kmer multiclass serotype accuracy when using a neural net
    data = acc(serotype_file, model='Neural network', data='Kmer',
               extra_col=['Kmer Length', 'Serotype'], extra_val=[7, 'all'])
    output['MultiWGSSerotypeNNAcc'] = data[0]


    # mean omnilog multiclasss serotype accuracy when using a neural net
    data = acc(serotype_file, model='Neural network', data='Omnilog',
               extra_col=['Serotype'], extra_val=['all'])
    output['MultiOmniSerotypeNNAcc'] = data[0]


    # Difference between kmer and omnilog accuracy when making o type predictions
    data1 = acc(otype_file, data='Kmer', extra_col=['Kmer Length'], extra_val=[7])
    data2 = acc(otype_file, data='Omnilog')
    output['DiffWGSOmniOtypeAcc'] = data1[0] - data2[0]


    # Difference between kmer and omnilog accuracy when making h type predictions
    data1 = acc(htype_file, data='Kmer', extra_col=['Kmer Length'], extra_val=[7])
    data2 = acc(htype_file, data='Omnilog')
    output['DiffWGSOmniHtypeAcc'] = data1[0] - data2[0]


    # Mean multiclass otype accuracy when using a neural net and both data types
    data = acc(otype_file, model='Neural network', extra_col=['Kmer Length', 'Otype'],
               extra_val=[7, 'all'])
    output['MultiOtypeNNAcc'] = data[0]


    # Mean multiclass otype accuracy when using a neural net and both data types
    data = acc(htype_file, model='Neural network', extra_col=['Kmer Length', 'Htype'],
               extra_val=[7, 'all'])
    output['MultiHtypeNNAcc'] = data[0]

#     data1 = acc(otype_file, model='Random forest', data='kmer')[0]
#     data1 -= acc(multiclass_file, model='Random forest', data='kmer', prediction='All Otype')[0]
# 
#     data2 = acc(htype_file, model='Random forest', data='omnilog')[0]
#     data2 -= acc(multiclass_file, model='Random forest', data='Omnilog', prediction='All Otype')[0]
# 
#     data3 = acc(otype_file, model='Support vector machine', data='Kmer')[0]
#     data3 -= acc(multiclass_file, model='Support vector machine', data='Kmer', prediction='All Htype')[0]
#     data4 = acc(htype_file, model='Support vector machine', data='Omnilog')[0]
#     data4 -= acc(multiclass_file, model='Support vector machine', data='Omnilog', prediction='All Htype')[0]
# 
#     data_min = min([data1, data2, data3, data4])
#     data_max = max([data1, data2, data3, data4])
#     output['MinMultiBinOHtypeRFSVMAcc'] = data_min
#     output['MaxMultiBinOHtypeRFSVMAcc'] = data_max


    # Mean kmer host accuracy: all models
    data = acc(host_file, data='Kmer', extra_col=['Kmer Length'], extra_val=[7])
    output['WGSHostAcc'] = data[0]


    # Mean omnilog host accuracy
    data = acc(host_file, data='omni', extra_col=['Kmer Length'], extra_val=[7])
    output['OmniHostAcc'] = data[0]


    # Maximum and minimum diffecerence between accuracies at predicting difference hosts
    data1 = acc(host_file, extra_col=['Host', 'Kmer Length'], extra_val=['Ovine', 7])[0]
    data2 = acc(host_file, extra_col=['Host', 'Kmer Length'], extra_val=['Water', 7])[0]
    data3 = acc(host_file, extra_col=['Host', 'Kmer Length'], extra_val=['Bovine', 7])[0]
    data4 = acc(host_file, extra_col=['Host', 'Kmer Length'], extra_val=['Human', 7])[0]
    output['MaxDiffSpecificHost'] = max(data1, data2) - min(data3, data4)
    output['MinDiffSpecificHost'] = min(data1, data2) - max(data3, data4)

    macros = ""
    for key, value in output.items():
        macros += '\\newcommand{{\\{}}}{{{:.2f}}}\n'.format(key, 100*value)
    with open(snakemake.output[0], 'w') as f:
        f.write(macros)

if __name__ == "__main__":
    main()
