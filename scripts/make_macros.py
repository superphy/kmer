import pickle

def make_header():
    output = '% This file was automatically generated using snakemake.\n'
    output += '% To verify how the numbers in this file were generated see: \n'
    output += '% superphy/kmer/scripts/values_for_paper.py and \n'
    output += '% superphy/kmer/scripts/make_macros.py\n\n'
    return output

def convert_data(data):
    output = ""
    for key, value in data.items():
        output += '\\newcommand{{\\{}}}{{{:.2f}}}\n'.format(key, value)
    return output

def main():
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    output_string = make_header()
    output_string += convert_data(data)
    with open(output_file, 'w') as f:
        f.write(output_string)

if __name__ == '__main__':
    main()
