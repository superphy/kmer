#!/usr/bin/env python

from Bio.Blast.Applications import NcbiblastnCommandline

blastn = NcbiblastnCommandline(query="../data/lspa6.fasta",
                               db="../data/o157",
                               outfmt=5,
                               out="../analyses/lspa6/lspa6.xml",
                               word_size=7)
blastn()


