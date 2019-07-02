attributes = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
omnilog_dataset = "omnilog"
kmer_dataset = "kmer"
rule all:
    input:
        expand("results/kmer_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", attribute = attributes, kmer_feat = kmer_feats),
        expand("results/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl", attribute = attributes, omnilog_feat = omnilog_feats)

rule kmer_split:
    input:
        expand("data/filtered/{attribute}/kmer_matrix.npy", attribute = attributes)
    output:
        "data/hyp_splits/kmer-{attribute}/splits/set{split}/"
    params:
        attribute = '{attribute}'
    shell:
        'python src/validation_split_hyperas.py kmer {params.attribute}'

rule omnilog_split:
    input:
        expand("data/filtered/{attribute}/omnilog_matrix.npy", attribute = attributes)
    output:
        "data/hyp_splits/omnilog-{attribute}/splits/set{split}/"
    params:
        attribute = '{attribute}'
    shell:
        'python src/validation_split_hyperas.py omnilog {params.attribute}'

rule kmer_hyperas:
    input:
        expand("data/hyp_splits/kmer-{attribute}/splits/set{split}/", attribute = attributes, split = splits)
    output:
        "data/kmer_{attribute}/{kmer_feat}feats_{split}.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        attribute = '{attribute}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.kmer_feat} {params.attribute} 10 {params.split} kmer'

rule omnilog_hyperas:
    input:
        expand("data/hyp_splits/omnilog-{attribute}/splits/set{split}/", attribute = attributes, split = splits)
    output:
        "data/omnilog_{attribute}/{omnilog_feat}feats_{split}.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.omnilog_feat} {params.attribute} 10 {params.split} omnilog'

rule kmer_average:
    input:
        expand("data/{kmer}_{attribute}/{kmer_feat}feats_{split}.pkl", attribute = attributes, split = splits, kmer_feat = kmer_feats, kmer = kmer_dataset)
    output:
        "results/{kmer}_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        attribute = '{attribute}',
        kmer = '{kmer}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} {params.attribute} {params.kmer}'

rule omnilog_average:
    input:
        expand("data/{omnilog}_{attribute}/{omnilog_feat}feats_{split}.pkl", attribute = attributes, split = splits, omnilog_feat = omnilog_feats, omnilog = omnilog_dataset)
    output:
        "results/{omnilog}_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        omnilog = '{omnilog}'
    shell:
        'python src/hyp_average.py {params.omnilog_feat} {params.attribute} {params.omnilog}'
