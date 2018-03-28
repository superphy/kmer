#!/usr/bin/env python
import re
import collections

def get_metadata():
    # metadata file
    nf_dict = {}
    with open("../data/ecomnilog_fasta_headers.txt", "r") as nfh:

        for l in nfh:
            m1 = re.search(r"==>\s([\w-]+)", l)
            if m1:
                filename = m1.group(1)
                ln = nfh.__next__()

                m2 = re.search(r">lcl\|([\w\-_]+)\|", ln)
                if m2:
                    fastaname = m2.group(1)
                    fn = fastaname.replace('-', '_')
                    nf_dict[fn] = filename
                else:
                    print("Can't match {}".format(l))
    return nf_dict


def get_panseq_names():
    pn_dict = {}
    with open("../analyses/panseq/phylip_name_conversion.txt", "r") as pfh:
        for l in pfh:
            la = l.split()
            m = re.search(r"\d+", l)
            if m:
                pn_dict[la[0]] = la[1]
    return pn_dict


def print_rosetta(pn, md):
    od = collections.OrderedDict(sorted(pn.items()))
    newfile = open("../data/phylip_rosetta.txt", "w")
    for k, v in od.items():
        newfile.write(str.join('\t', (k, v, md[v], "\n")))


if __name__ == "__main__":
    md = get_metadata()
    pn = get_panseq_names()
    print_rosetta(pn, md)
    print("All done")