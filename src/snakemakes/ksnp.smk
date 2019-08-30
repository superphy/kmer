""" Waffles
sbatch -c 32 --mem 64G snakemake -j 16 -s src/ksnp3.smk
"""

""" kSNP3
documentation: http://gensoft.pasteur.fr/docs/kSNP3/01/kSNP3.01%20User%20Guide%20.pdf
1. download https://sourceforge.net/projects/ksnp/
2. $ sudo apt-get install tcsh
    > you can check what shell is running with $ echo $0
3. extract the folder; place the kSNP3 folder in desired spot (and keep it there)
        a. default location is the /usr/local directory
            > on my home computer it's ~/../../user/local/
            > $ sudo mv kSNP3 </usr/local>
        b. if you use a different location
            > in the kSNP3 folder, open the kSNP file and edit the path at the top
4. add the location to PATH
    > can check path with $ PATH
    > I had annoying path problems so in my snakemake I do
        home: $ "export PATH='usr/local/kSNP3:$PATH' && <ksnp command>"
        wafs: $ "export PATH='~/kSNP3.1_Linux_package/kSNP3:$PATH"
    > I plan to troubleshoot this later
"""

ENA_RAW = "data/genomes/raw/"
ENA_CLEAN = "data/genomes_clean/ENA/"
ENA_ids,ENA_prefixes = glob_wildcards(ENA_RAW+"{ENA_id}/{ENA_prefix}_1.fastq.gz")

FLAG = "data/flags/" # location for assembly complete flags
ASMBL = "data/assemblies/" # location of assembled genomes
# home
KSNP_PATH = '/home/gates/Desktop/kSNP3'
KSNP_CPU = '7'
KMER_SIZE = '11'
MECOLI_DIR = '/home/gates/Desktop/kmer/'
'''
# waffles
KSNP_PATH = '~/kSNP3.1_Linux_package/kSNP3:$PATH'
KSNP_CPU = '32'
KMER_SIZE = '11'
MECOLI_DIR = '/mecoli/'
'''
rule all:
    input:
        "data/kSNP3_"+KMER_SIZE+"mer/Logfile.txt"
        #"data/kSNP3_output/Logfile.txt"
rule add_header_id:
    input:
        ASMBL+"{GROUP}/{id}/spades/scaffolds.fasta"
    output:
        ASMBL+"{GROUP}/{id}/spades/scaffolds_header_{prefix}.fasta"
    run:
        pre = wildcards.prefix
        rename_header(input,output,pre)
"""
adapted from https://github.com/superphy/AMR_Predictor/blob/master/src/master_fasta.py
"""
def rename_header(fasta_in, fasta_out, prefix):
    from Bio import Seq, SeqIO
    fasta_in = str(fasta_in)
    fasta_out = str(fasta_out)
    for record in SeqIO.parse(fasta_in, "fasta"):
        contig_seq = record.seq
        contig_seq = contig_seq._get_seq_str_and_check_alphabet(contig_seq)
        contig_header = record.id
        #file_name = fasta_in.split('/')[-1].split('.')[0]
        prefix = str(prefix)
        contig_header = ">{}_{}".format(prefix, contig_header)
        with open(fasta_out,'a') as fout:
            fout.write(contig_header)
            fout.write("\n")
            fout.write(contig_seq)
            fout.write("\n")
    return 0
rule in_file:
    input:
        ENA = expand(ASMBL+"{GROUP}/{ENA_id}/spades/scaffolds_header_{ENA_prefix}.fasta", zip, GROUP=["ENA"]*len(ENA_ids), ENA_id=ENA_ids, ENA_prefix=ENA_prefixes)
        #ENA = expand(ASMBL+"{GROUP}/{ENA_id}/spades/scaffolds_header_{ENA_prefix}.fasta", zip, GROUP="ENA", ENA_id=ENA_ids, ENA_prefix=ENA_prefixes)
    output:
        "data/in_list.txt"
    run:
        # write to a file
        # for each input, add it to the file, new line, loop
        # format for each line is 'path/to/file.fa [tab] prefix'
        with open(output[0], "a") as out:
            for fpath in input:
                id=fpath.split('scaffolds_header_')[-1].split('.fasta')[0]
                #nline = '\n'.join(['{}\t{}\n'.format(fpath,id)])
                test = '\n'.join(['{}{}\t{}\n'.format(MECOLI_DIR,fpath,id)])
                #out.write(nline)
                out.write(test)
rule ksnp3:
    input:
        "data/in_list.txt"
    output:
        #"data/kSNP3_output/Logfile.txt"
        "data/kSNP3_"+KMER_SIZE+"mer/Logfile.txt"
    params:
        #outdir = "data/kSNP3_output/",
        outdir = "data/kSNP3_"+KMER_SIZE+"mer/",
        #ksnp_waf = '~/kSNP3.1_Linux_package/kSNP3:$PATH',
        #ksnp_home = '~/../../usr/local/kSNP3:$PATH'
    shell:
        #"export PATH={params.ksnp_home}\
        "export PATH={KSNP_PATH}\
        && \
        kSNP3 -in data/in_list.txt -outdir {params.outdir} -k {KMER_SIZE} -CPU {KSNP_CPU} | tee {params.outdir}Logfile.txt"
