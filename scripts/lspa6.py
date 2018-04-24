#!/usr/bin/env python

from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
from collections import defaultdict
import re
import sys



def run_blast():
    """
    Query the O157 database with the lspa6 markers.
    Generate the blast xml output.
    :return: blast output xml
    """
    output_xml = "../analyses/lspa6/lspa6.xml"
    blastn = NcbiblastnCommandline(query="../data/lspa6.fasta",
                                   db="../data/o157",
                                   outfmt=5,
                                   out=output_xml,
                                   word_size=7,
                                   gapopen=1,
                                   gapextend=1,
                                   max_hsps=1,
                                   )
    blastn()
    return output_xml


def parse_blast(xml_file):
    """
    Get the LSPA6 designations from the results
    :param xml_file: output of LSPA6 fasta markers against the O157 strains in this study
    :return: Dictionary of genome names and LSPA6 types
    """
    lspa_dict = defaultdict(lambda: defaultdict(str))

    with open(xml_file) as xml_fh:
        blast_records = NCBIXML.parse(xml_fh)
        for r in blast_records:
            q = r.query
            m = re.match(r"(\S+)", q)
            query = None
            if m:
                query = m.group(1)
            else:
                print("No query")
                exit(1)

            for hit in r.alignments:
                hname = get_rosetta_name(hit.hit_def)
                ltype = get_ltype(query, hit)
                lspa_dict[hname][query] = ltype

    return lspa_dict


def get_rosetta_name(rawname):
    """
    Take the raw name from blast and return a clean, standardized name, the same as used for building the tree, using the same "rosetta" file.
    :param rawname: Fasta header name from the blast results
    :return: The cleaned name based on the phylip_rossetta.txt file
    """
    m = re.match(r"lcl\|([[a-zA-Z0-9\-]+)", rawname)
    if m:
        return m.group(1)
    else:
        print("No rosetta name found")
        exit(1)


def get_ltype(gene, hit):
    """
    Based on the gene and the length, assing 1 or 2 to the allele
    :param gene: the LSPA6 gene used as query for the given alignment
    :param hit: Blast hit for the gene and the genome
    :return: the 1 or 2 lineage designation for the gene
    """

    lsizes = {
        "folD": 161,
        "Z5935": 133,
        "yhcG": 394,
        "rbsB": 218,
        "rtcB": 270,
        "arp": 315
    }

    ltype = "2"
    hsp = hit.hsps[0]
    hlength = hsp.align_length
    difference = abs(lsizes[gene] - hlength)

    if difference < 1:
        ltype = "1"
    else:
        print("Gene {} has a difference of {}".format(gene, difference), file= sys.stderr)
    return ltype


def get_formatted_results(res):
    """
    Takes in a result dictionary, returns formatted list of LSPA6 type
    :param res: Dictionary of genome->gene->type
    :return: A list of one line per genome with LSPA6 type
    """

    lspa6_results = []
    for k,v in res.items():
        lspa6_results.append(k + " " + v["folD"] + v["Z5935"] + v["yhcG"] + v["rtcB"] + v["rbsB"] + v["arp"])

    return lspa6_results


if __name__ == "__main__":
    blast_xml = run_blast()
    results = parse_blast(blast_xml)
    print("\n".join(get_formatted_results(results)))
