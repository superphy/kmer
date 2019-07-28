attributes = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
ranges = [i for i in range(1, 2, 1)]
omnilog_dataset = "omnilog"
kmer_dataset = "kmer"
rule all:
    input:
        expand("data{range}/kmer_{attribute}/{kmer_feat}feats_{split}.pkl", attribute = attributes, kmer_feat = kmer_feats, range = ranges, split = splits),
        expand("data{range}/omnilog_{attribute}/{omnilog_feat}feats_{split}.pkl", attribute = attributes, omnilog_feat = omnilog_feats, range = ranges, split = splits)

rule kmer_split:
    input:
        "data/filtered/{attribute}/kmer_matrix.npy"
    output:
        "data{range}/hyp_splits/kmer-{attribute}/splits/set{split}/"
    params:
        attribute = '{attribute}',
        range = '{range}'
    shell:
        'python src/validation_split_hyperas.py kmer {params.attribute} {params.range}'

rule omnilog_split:
    input:
        "data/filtered/{attribute}/omnilog_matrix.npy"
    output:
        "data{range}/hyp_splits/omnilog-{attribute}/splits/set{split}/"
    params:
        attribute = '{attribute}',
        range = '{range}'
    shell:
        'python src/validation_split_hyperas.py omnilog {params.attribute} {params.range}'

rule kmer_hyperas:
    input:
        "data{range}/hyp_splits/kmer-{attribute}/splits/set{split}/"
    output:
        "data{range}/kmer_{attribute}/{kmer_feat}feats_{split}.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        attribute = '{attribute}',
        split = '{split}',
        range = '{range}'
    shell:
        'python src/hyp.py {params.kmer_feat} {params.attribute} 10 {params.split} kmer {params.range}'

rule omnilog_hyperas:
    input:
        "data{range}/hyp_splits/omnilog-{attribute}/splits/set{split}/"
    output:
        "data{range}/omnilog_{attribute}/{omnilog_feat}feats_{split}.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        split = '{split}',
        range = '{range}'
    shell:
        'python src/hyp.py {params.omnilog_feat} {params.attribute} 10 {params.split} omnilog {params.range}'
