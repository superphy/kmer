
#################################################################

# Location of the raw genomes
RAW_GENOMES_PATH = "data/genomes/uk_us/"

# Kmer length that you want to count
KMER_SIZE = 11

# Data type of the resulting kmer matrix. Use uint8 if counts are
# all under 256. Else use uint16 (kmer counts under 65536)
MATRIX_DTYPE = 'uint8'

#################################################################

ids, = glob_wildcards(RAW_GENOMES_PATH+"{id}.fasta")

rule all:
  input:
    ".uk_us_touchfile.txt"
"""
rule clean:
  input:
    RAW_GENOMES_PATH+"{id}.fasta"
  output:
    "data/genomes/clean/{id}.fasta"
  run:
    shell("python src/clean.py {input} data/genomes/clean/")
"""
rule count:
  input:
    RAW_GENOMES_PATH+"{id}.fasta"
  output:
    temp("data/uk_us_jellyfish_results/{id}.jf")
  threads:
    2
  shell:
    "jellyfish count -C -m {KMER_SIZE} -s 100M -t {threads} {input} -o {output}"

rule dump:
    input:
        "data/uk_us_jellyfish_results/{id}.jf"
    output:
        "data/uk_us_jellyfish_results/{id}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule matrix:
    input:
        expand("data/uk_us_jellyfish_results/{id}.fa", id=ids)
    output:
        touch(".uk_us_touchfile.txt")
    shell:
        "python src/parallel_matrix.py {KMER_SIZE} {MATRIX_DTYPE} data/uk_us_jellyfish_results/ data/uk_us_unfiltered/"
