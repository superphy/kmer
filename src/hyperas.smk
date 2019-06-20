attributes = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
rule all:
    input:
        expand("results/kmer_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", attribute = attributes, kmer_feat = kmer_feats),
        expand("results/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnOmnilog_testedOnaCrossValidation.pkl", attribute = attributes, omnilog_feat = omnilog_feats)

rule kmer_split:
    input:
        expand("data/filtered/{attribute}/kmer_matrix.npy", attribute = attributes)
    output:
        "data/hyp_splits/kmer-{attribute}/splits/set1/"
    shell:
        'python src/validation_split_hyperas.py kmer {attributes}'

rule omnilog_split:
    input:
        expand("data/filtered/{attribute}/omnilog_matrix.npy", attribute = attributes)
    output:
        "data/hyp_splits/omnilog-{attribute}/splits/set1/"
    shell:
        'python src/validation_split_hyperas.py omnilog {attributes}'

rule kmer_hyperas:
    input:
        expand("data/hyp_splits/kmer-{attribute}/splits/set1/", attribute = attributes)
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
        expand("data/hyp_splits/omnilog-{attribute}/splits/set1/", attribute = attributes)
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
        expand("data/kmer_{attribute}/{kmer_feat}feats_{split}.pkl", attribute = attributes, split = splits, kmer_feat = kmer_feats)
    output:
        "results/kmer_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        attribute = '{attribute}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} {params.attribute} kmer'

rule omnilog_average:
    input:
        expand("data/omnilog_{attribute}/{omnilog_feat}feats_{split}.pkl", attribute = attributes, split = splits, omnilog_feat = omnilog_feats)
    output:
        "results/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnOmnilog_testedOnaCrossValidation.pkl"
    shell:
        'python src/hyp_average.py {params.omnilog_feat} {params.attribute} omnilog'
