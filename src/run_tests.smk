
rule all:
    input:
        "results/kmer_serotype",
        "results/omnilog_serotype",
        "results/kmer_host",
        "results/omnilog_host",
        "results/uk_host",
        "results/us_host",
        "results/ukus_host",
        "results/uk2us_host",
        "results/us2uk_host",
        "results/kmer2ukus_host",
        "results/kmer_otype",
        "results/kmer_htype",
        "results/omnilog_otype",
        "results/omnilog_htype"
    shell:
        "echo All tests deployed"

rule kmer_serotype:
    output:
        "results/kmer_serotype"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kmer -a Serotype -o {output} -f $i -m $j; done; done'

rule omnilog_serotype:
    output:
        "results/omnilog_serotype"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 10 10 190); do python src/model.py -x omnilog -a Serotype -o {output} -f $i -m $j; done; done'

rule kmer_host:
    output:
        "results/kmer_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kmer -a Host -o {output} -f $i -m $j; done; done'

rule omnilog_host:
    output:
        "results/omnilog_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 10 10 190); do python src/model.py -x omnilog -a Host -o {output} -f $i -m $j; done; done'

rule uk_host:
    output:
        "results/uk_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x uk -a Host -o {output} -f $i -m $j; done; done'

rule us_host:
    output:
        "results/us_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x us -a Host -o {output} -f $i -m $j; done; done'

rule ukus_host:
    output:
        "results/ukus_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x uk_us -a Host -o {output} -f $i -m $j; done; done'

rule us2uk_host:
    output:
        "results/us2uk_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x us -y uk -a Host -o {output} -f $i -m $j; done; done'

rule uk2us_host:
    output:
        "results/uk2us_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x uk -y us -a Host -o {output} -f $i -m $j; done; done'

rule kmer2ukus_host:
    output:
        "results/kmer2ukus_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kmer -y uk_us -a Host -o {output} -f $i -m $j; done; done'

rule ukus2kmer_host:
    output:
        "results/ukus2kmer_host"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x uk_us -y kmer -a Host -o {output} -f $i -m $j; done; done'

rule kmer_otype:
    output:
        "results/kmer_otype"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kmer -a Otype -o {output} -f $i -m $j; done; done'

rule kmer_htype:
    output:
        "results/kmer_htype"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kmer -a Htype -o {output} -f $i -m $j; done; done'


rule omnilog_otype:
    output:
        "results/omnilog_otype"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 10 10 190); do python src/model.py -x omnilog -a Otype -o {output} -f $i -m $j; done; done'

rule omnilog_htype:
    output:
        "results/omnilog_htype"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 10 10 190); do python src/model.py -x omnilog -a Htype -o {output} -f $i -m $j; done; done'
