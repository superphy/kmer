#!/usr/bin/env python

from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
from collections import defaultdict
import re



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
                                   word_size=7)
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
                hname = hit.hit_def
                ltype = get_ltype(query, hit)
                lspa_dict[hit][hname] = ltype

    return lspa_dict


def get_ltype(gene, hit):
    """
    Based on the gene and the length, assing 1 or 2 to the allele
    :param gene: the LSPA6 gene used as query for the given alignment
    :param hit: Blast hit for the gene and the genome
    :return: the 1 or 2 lineage designation for the gene
    """

    ltype = "2"
    hsp = hit.hsps[0]
    hlength = hsp.positives
    if gene == "folD":
        if hlength  == 161:
            ltype = "1"
    elif gene == "Z5935":
        if hlength == 133:
            ltype = "1"
    elif gene == "yhcG":
        if hlength == 394:
            ltype = "1"
    elif gene == "rbsB":
        if hlength == 218:
            ltype = "1"
    elif gene == "rtcB":
        if hlength == 270:
            ltype = "1"
    elif gene == "arp":
        if hlength == 315:
            ltype = "1"
    else:
        print("Incorrect gene for LSPA6 prediction: {}".format(gene))
        exit(1)

    return ltype


if __name__ == "__main__":
    blast_xml = run_blast()
    results = parse_blast(blast_xml)
    print(results)