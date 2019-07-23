attributes = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
ranges = [i for i in range(1, 51, 1)]
omnilog_dataset = "omnilog"
kmer_dataset = "kmer"

rule all:
    input:
        expand("results{range}/{kmer}_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", kmer = kmer_dataset, attribute = attributes, kmer_feat = kmer_feats, range = ranges),
        expand("results{range}/{omnilog}_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl",omnilog = omnilog_dataset, attribute = attributes, omnilog_feat = omnilog_feats, range = ranges)
rule kmer_average:
    input:
        expand("data{range}/{kmer}_{attribute}/{kmer_feat}feats_{split}.pkl", kmer = kmer_dataset, attribute = attributes, kmer_feat = kmer_feats, range = ranges, split = splits)
    output:
        "results{range}/{kmer}_{attribute}/{attribute}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        attribute = '{attribute}',
        kmer = '{kmer}',
        range = '{range}',
        split = '{split}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} {params.attribute} {params.kmer} {params.range}'

rule omnilog_average:
    input:
        expand("data{range}/{omnilog}_{attribute}/{omnilog_feat}feats_{split}.pkl",omnilog = omnilog_dataset, attribute = attributes, omnilog_feat = omnilog_feats, range = ranges, split = splits)
    output:
        "results{range}/{omnilog}_{attribute}/{attribute}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        attribute = '{attribute}',
        omnilog = '{omnilog}',
        range = '{range}',
        split = '{split}'
    shell:
        'python src/hyp_average.py {params.omnilog_feat} {params.attribute} {params.omnilog} {params.range}'
