attributes = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
rule all:
    input:
        expand("results/kmer_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", attribute = attributes, kmer_feat = kmer_feats),
        expand("results/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnOmnilog_testedOnaCrossValidation.pkl", attribute = attributes, omnilog_feat = omnilog_feats)

rule split:
    input:
        expand("data/filtered/{attribute}/kmer_matrix.npy", attribute = attributes),
        expand("data/filtered/{attribute}/omnilog_matrix.npy", attribute = attributes)
    output:
        "data/hyp_splits/kmer-{attribute}/splits/set1/",
        "data/hyp_splits/omnilog-{attribute}/splits/set1/"
    shell:
        'python src/validation_split_hyperas.py kmer {attributes}'
        'python src/validation_split_hyperas.py omnilog {attributes}'

rule hyperas:
    input:
        expand("data/hyp_splits/kmer-{attribute}/splits/set1/", attribute = attributes),
        expand("data/hyp_splits/omnilog-{attribute}/splits/set1/", attribute = attributes)
    output:
        "data/kmer_{attribute}/{kmer_feat}feats_{split}.pkl",
        "data/omnilog_{attribute}/{omnilog_feat}feats_{split}.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.kmer_feat} {params.attribute} 10 {params.split} kmer',
        'python src/hyp.py {params.omnilog_feat} {params.attribute} 10 {params.split} omnilog'

rule average:
    input:
        expand("data/kmer_{attribute}/{feat}feats_{split}.pkl", attribute = attributes, split = splits, kmer_feat = kmer_feats),
        expand("data/omnilog_{attribute}/{feat}feats_{split}.pkl", attribute = attributes, split = splits, omnilog_feat = omnilog_feats)
    output:
        "results/kmer_{attribute}/{attribute}_{feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl",
        "results/omnilog_{attribute}/{attribute}_{feat}feats_ANNtrainedOnOmnilog_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} {params.attribute} kmer',
        'python src/hyp_average.py {params.omnilog_feat} {params.attribute} omnilog'
