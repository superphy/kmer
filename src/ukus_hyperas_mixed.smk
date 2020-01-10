splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]

rule all:
    input:
        expand("results/uk2us_host/host_{kmer_feat}feats_ANNtrainedOnuk_testedOnus.pkl", kmer_feat = kmer_feats),
        expand("results/us2uk_host/host_{kmer_feat}feats_ANNtrainedOnus_testedOnuk.pkl", kmer_feat = kmer_feats)

rule uk2us_host_split:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "data/hyp_splits/uk-host/splits/set{split}/"
    shell:
        'python src/validation_split_hyperas.py uk host'

rule uk2us_host_hyperas:
    input:
        expand("data/hyp_splits/uk-host/splits/set{split}/", split = splits)
    output:
        "data/uk_host/{kmer_feat}feats_{split}_ustest.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        split = '{split}'
    shell:
        'python src/hyp_mixed.py {params.kmer_feat} host 10 {params.split} uk us'

rule uk2us_host_average:
    input:
        expand("data/uk_host/{kmer_feat}feats_{split}_ustest.pkl", split = splits, kmer_feat = kmer_feats)
    output:
        "results/uk2us_host/host_{kmer_feat}feats_ANNtrainedOnuk_testedOnus.pkl"
    params:
        kmer_feat = '{kmer_feat}'
    shell:
        'python src/hyp_average_mixed.py {params.kmer_feat} host uk us'

rule us2uk_host_split:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "data/hyp_splits/us-host/splits/set{split}/"
    shell:
        'python src/validation_split_hyperas.py us host'

rule us2uk_host_hyperas:
    input:
        expand("data/hyp_splits/us-host/splits/set{split}/", split = splits)
    output:
        "data/us_host/{kmer_feat}feats_{split}_uktest.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        split = '{split}'
    shell:
        'python src/hyp_mixed.py {params.kmer_feat} host 10 {params.split} us uk'

rule us2uk_host_average:
    input:
        expand("data/us_host/{kmer_feat}feats_{split}_uktest.pkl", split = splits, kmer_feat = kmer_feats)
    output:
        "results/us2uk_host/host_{kmer_feat}feats_ANNtrainedOnus_testedOnuk.pkl"
    params:
        kmer_feat = '{kmer_feat}'
    shell:
        'python src/hyp_average_mixed.py {params.kmer_feat} host us uk'
