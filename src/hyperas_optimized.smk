attributes = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
ranges = [i for i in range(1, 20, 1)]
omnilog_dataset = "omnilog"
kmer_dataset = "kmer"
rule all:
    input:
        expand("results{range}/kmer_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", attribute = attributes, kmer_feat = kmer_feats, range = ranges),
        expand("results{range}/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl", attribute = attributes, omnilog_feat = omnilog_feats, range = ranges)

rule kmer_split:
    input:
        expand("data/filtered/{attribute}/kmer_matrix.npy", attribute = attributes)
    output:
        "data{range}/hyp_splits/kmer-{attribute}/splits/set{split}/"
    params:
        attribute = '{attribute}',
        range = '{range}',
        split = '{split}'
    shell:
        'python src/validation_split_hyperas.py kmer {params.attribute} {params.range}'

rule omnilog_split:
    input:
        expand("data/filtered/{attribute}/omnilog_matrix.npy", attribute = attributes)
    output:
        "data{range}/hyp_splits/omnilog-{attribute}/splits/set{split}/"
    params:
        attribute = '{attribute}',
        range = '{range}',
        split = '{split}'
    shell:
        'python src/validation_split_hyperas.py omnilog {params.attribute} {params.range}'

rule kmer_hyperas:
    input:
        rules.kmer_split.output
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
        rules.omnilog_split.output
    output:
        "data{range}/omnilog_{attribute}/{omnilog_feat}feats_{split}.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        split = '{split}',
        range = '{range}'
    shell:
        'python src/hyp.py {params.omnilog_feat} {params.attribute} 10 {params.split} omnilog {params.range}'

rule kmer_average:
    input:
        rules.kmer_hyperas.output
    output:
        "results{range}/{kmer}_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        attribute = '{attribute}',
        kmer = '{kmer}',
        range = '{range}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} {params.attribute} {params.kmer} {params.range}'

rule omnilog_average:
    input:
        rules.omnilog_hyperas.output
    output:
        "results{range}/{omnilog}_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        omnilog = '{omnilog}',
        range = '{range}'
    shell:
        'python src/hyp_average.py {params.omnilog_feat} {params.attribute} {params.omnilog} {params.range}'
