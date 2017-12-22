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
    x = [float(d[0]) for d in data if d]
    try:
        y = [float(d[1]) for d in data if d]
    except IndexError as E:
        print E
        return
    return x,y

def main(filepath, title, xlabel, ylabel, x_range, enforce_x, y_range, enforce_y, keep, log, add_trends):
    plt.figure()
    if title:
        plt.title(' '.join(title))
    if x_range:
        plt.xticks(np.arange(x_range[0], x_range[1], x_range[2]))
        if enforce_x:
            plt.xlim((x_range[0], x_rangep[1]))
    if log:
        plt.xscale('log', basex=2)
    if y_range:
        plt.yticks(np.arange(y_range[0], y_range[1], y_range[2]))
        if enforce_y:
            plt.ylim((y_range[0], y_range[1]))
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
	#      files = sorted(files, key=lambda x: int(filter(str.isdigit, x)))
	count = 1
    for f in files:
        data = get_data(f)
        if data:
            if count > 10:
                line_style = '--'
            else:
                line_style = '-'
            count += 1
            p = plt.plot(data[0], data[1], label=f.replace('.txt', '').replace(filepath, ''), linestyle=line_style)
            if add_trends:
                colour = p[-1].get_color()
                coefs = np.polyfit(np.asarray(data[0], dtype='float64'), data[1], add_trends)
                func = np.poly1d(coefs)
                y = func(data[0])
                plt.plot(data[0], y, '--', color=colour)


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
    parser.add_argument('--enforce_x', '-ex',
                        help='Treated as a bool. Force the max/min value of x axis to match given x_range',
                        type=bool)
    parser.add_argument('--y_range', '-yr',
                        help='start,stop,step for the location of y tick marks',
                        nargs=3,
                        type=float)
    parser.add_argument('--enforce_y', '-ey',
                        help='Treated as a bool. Force the max/min value of y axis to match given y_range',
                        type=bool)
    parser.add_argument('--keep', '-k',
                        help='A substring that must be included in the filenames for them to be plotted')
    parser.add_argument('--log', '-l',
	                    help='If true the x axis is plotted in log space',
	                    type=bool)
    parser.add_argument('--add_trends', '-at',
                        help='Int, polynomial trendlines of degree given are plotted',
                        type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
