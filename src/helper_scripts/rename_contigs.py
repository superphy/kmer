import os, sys
from Bio import SeqIO

def update(directory, file):
    path = directory + file
    temp = []
    for seq_record in SeqIO.parse(path, "fasta"):
        local = "lcl|{}|".format(seq_record.id)
        seq_record.id = local
        description = seq_record.description
        description = description.replace("-", "_")
        description = description.replace(" ", "_")
        description = description.replace(".", "_")
        seq_record.description = description
        temp.append(seq_record) # list of contigs in a fasta file
    SeqIO.write(temp, "data/genomes/renamed_new/"+file, "fasta") # write contigs to fasta

if __name__ == "__main__":
    directory = sys.argv[1]

    for file in os.listdir(directory):
        update(directory, file)
        
