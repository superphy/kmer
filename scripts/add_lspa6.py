#!/usr/bin/env python
import csv
from collections import defaultdict


def print_metadata(md):
    # metadata file
    with open("../data/omnilog_metadata.csv", "r") as metadata_file:
            metadata_fh = csv.reader(metadata_file, delimiter=',')
            first_line = next(metadata_fh)
            print(','.join(first_line) + ',LSPA6')
            for row in metadata_fh:
                ls = 'NA'
                if md[row[0]]:
                    ls = md[row[0]]

                print(','.join(row) + ',' + ls)


def get_lspa6_types():
    lspa_dict = defaultdict(str)
    with open("../analyses/lspa6_types.txt", "r") as lspa_file:
        lspa_fh = csv.reader(lspa_file, delimiter=' ')
        for row in lspa_fh:
            lspa_dict[row[0]]=row[1]

    return lspa_dict


if __name__ == "__main__":
    md = get_lspa6_types()
    print_metadata(md)