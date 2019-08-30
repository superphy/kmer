from Bio import Seq, SeqIO
import os
import subprocess

path = '/home/gates/Desktop/kmer/data/genomes/raw/'

with open("data/in_list.txt", 'a') as out:
    for file in os.listdir(path):
        id = file.split('.')
        id = id[0]
        line = path + file + '\t' + id + '\n'
        out.write(line)
