import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
import os

def get_data(filename):
    try:
        with open(filename, 'r') as f:
            data = f.read()
    except IOError as E:
        print E
        return
    data = data.split('\n')
    data = [x.split(',') for x in data if x]
    x = [d[0] for d in data if d]
    try:
        y = [d[1] for d in data if d]
    except IndexError as E:
        print E
        return
    return x,y

def main(filepath, title, xlabel, ylabel, x_range, y_range, keep, log):
    plt.figure()
    if title:
        plt.title(' '.join(title))
    if x_range:
        plt.xticks(np.arange(x_range[0], x_range[1], x_range[2]))
    if log:
        plt.xscale('log', basex=2)
    if y_range:
        plt.yticks(np.arange(y_range[0], y_range[1], y_range[2]))
    plt.grid()
    if xlabel:
        plt.xlabel(' '.join(xlabel))
    if ylabel:
        plt.ylabel(' '.join(ylabel))
    if keep:
        valid = [x for x in os.listdir(filepath) if keep in x]
        files = [filepath + x for x in valid]
    else:
        files = [filepath + x for x in os.listdir(filepath) if '.py' not in x]
    for f in files:
        data = get_data(f)
        if data:
            plt.plot(data[0], data[1], label=f.replace('.txt', '').replace(filepath, ''))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-f',
                        help='Path to files that you want ot plot',
                        required=True)
    parser.add_argument('--title', '-t',
                        help='The title of the figure',
                        nargs='*')
    parser.add_argument('--xlabel', '-xl',
                        help='The label for the x axis',
                        nargs='*')
    parser.add_argument('--ylabel', '-yl',
                        help='The label for the y axis',
                        nargs='*')
    parser.add_argument('--x_range', '-xr',
                        help='start,stop,step for the location of x tick marks',
                        nargs=3,
                        type=float)
    parser.add_argument('--y_range', '-yr',
                        help='start,stop,step for the location of y tick marks',
                        nargs=3,
                        type=float)
    parser.add_argument('--keep', '-k',
                        help='A substring that must be included in the filenames for them to be plotted')
    parser.add_argument('--log', '-l',
	                    help='If true the x axis is plotted in log space',
	                    type=bool)

    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
