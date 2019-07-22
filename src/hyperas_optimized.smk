groups = ["Host", "Serotype", "Otype", "Htype"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
ranges = [i for i in range(1, 21, 1)]

rule all:
    input:
        expand("results{range}/kmer_{group}/{group}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", group = groups, kmer_feat = kmer_feats, range = ranges),
        expand("results{range}/omnilog_{group}/{group}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl", group = groups, omnilog_feat = omnilog_feats, range = ranges)

rule kmer_split:
    input:
        expand("data/filtered/{group}/kmer_matrix.npy", group = groups)
    output:
        expand("data{range}/hyp_splits/kmer-{group}/splits/set{split}/", range = ranges, group = groups, split = splits)
    run:
        shell('python src/validation_split_hyperas.py kmer {wildcards.group} {wildcards.range}')

rule omnilog_split:
    input:
        expand("data/filtered/{group}/omnilog_matrix.npy", group = groups)
    output:
        expand("data{range}/hyp_splits/omnilog-{group}/splits/set{split}/", range = ranges, group = groups, split = splits)
    shell:
        'python src/validation_split_hyperas.py omnilog {wildcards.group} {wildcards.range}'

rule kmer_hyperas:
    input:
        rules.kmer_split.output
    output:
        expand("data{range}/kmer_{group}/{kmer_feat}feats_{split}.pkl", range = ranges, group = groups, kmer_feat = kmer_feats, split = splits)
    shell:
        'python src/hyp.py {wildcards.kmer_feat} {wildcards.group} 10 {wildcards.split} kmer {wildcards.range}'

rule omnilog_hyperas:
    input:
        rules.omnilog_split.output
    output:
        expand("data{range}/omnilog_{group}/{omnilog_feat}feats_{split}.pkl", range = ranges, group = groups, omnilog_feat = omnilog_feats, split = splits)
    shell:
        'python src/hyp.py {wildcards.omnilog_feat} {wildcards.group} 10 {wildcards.split} omnilog {wildcards.range}'

rule kmer_average:
    input:
        rules.kmer_hyperas.output
    output:
        expand("results{range}/kmer_{group}/{group}_{kmer_feat}feats_ANNtrainedOnkmer_testedOnaCrossValidation.pkl", range = ranges, group = groups, kmer_feat = kmer_feats)
    shell:
        'python src/hyp_average.py {wildcards.kmer_feat} {wildcards.group} {wildcards.kmer} {wildcards.range}'

rule omnilog_average:
    input:
        rules.omnilog_hyperas.output
    output:
        expand("results{range}/omnilog_{group}/{group}_{omnilog_feat}feats_ANNtrainedOnomnilog_testedOnaCrossValidation.pkl", range = ranges, group = groups, omnilog_feat = omnilog_feats)
    shell:
        'python src/hyp_average.py {wildcards.omnilog_feat} {wildcards.group} {wildcards.omnilog} {wildcards.range}'
