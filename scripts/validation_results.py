from omni_results import acc

def main():
    validation_file = snakemake.input[0]

    output = {}

    data = acc(validation_file, data='Kmer', model='Neural network',
               extra_col=['Dataset', 'Kmer Length', 'Kmer Filter'],
               extra_val=['Split', 7, 'Complete'])
    output['ValidationNNSplit'] = data[0]

    data = acc(validation_file, data='Kmer', model='Neural network',
               extra_col=['Dataset', 'Kmer Length', 'Kmer Filter'],
               extra_val=['Mixed', 7, 'Complete'])
    output['ValidationNNMixed'] = data[0]

    data = acc(validation_file, data='Kmer', model='Support vector machine',
               extra_col=['Dataset', 'Kmer Length', 'Kmer Filter'],
               extra_val=['Split', 7, 'Complete'])
    output['ValidationSVMSplit'] = data[0]

    data = acc(validation_file, data='Kmer', model='Support vector machine',
               extra_col=['Dataset', 'Kmer Length', 'Kmer Filter'],
               extra_val=['Mixed', 7, 'Complete'])
    output['ValidationSVMMixed'] = data[0]

    macros = ""
    for key, value in output.items():
        macros += '\\newcommand{{\\{}}}{{{:.2f}}}\n'.format(key, value*100)
    with open(snakemake.output[0], 'w') as f:
        f.write(macros)

if __name__ == "__main__":
    main()
