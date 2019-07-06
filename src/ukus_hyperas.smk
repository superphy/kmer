splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]

rule all:
    input:
        expand("results/uk_host/host_{kmer_feat}feats_ANNtrainedOnuk_testedOnaCrossValidation.pkl", kmer_feat = kmer_feats),
        #expand("results/uk2us_host/Host_{kmer_feat}feats_ANNtrainedOnuk_testedOnus.pkl", kmer_feat = kmer_feats),
        expand("results/uk_us_host/host_{kmer_feat}feats_ANNtrainedOnuk_us_testedOnaCrossValidation.pkl", kmer_feat = kmer_feats),
        #expand("results/ukus2kmer_host/Host_{kmer_feat}feats_ANNtrainedOnukus_testedOnkmer.pkl", kmer_feat = kmer_feats),
        expand("results/us_host/host_{kmer_feat}feats_ANNtrainedOnus_testedOnaCrossValidation.pkl", kmer_feat = kmer_feats),
        #expand("results/us2uk_host/Host_{kmer_feat}feats_ANNtrainedOnus_testedOnuk.pkl", kmer_feat = kmer_feats),
        #expand("results/kmer2ukus_host/Host_{kmer_feat}feats_ANNtrainedOnkmer_testedOnukus.pkl", kmer_feat = kmer_feats)

rule uk_host_split:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "data/hyp_splits/uk-host/splits/set{split}/"
    shell:
        'python src/validation_split_hyperas.py uk host'

rule uk_host_hyperas:
    input:
        expand("data/hyp_splits/uk-host/splits/set{split}/", split = splits)
    output:
        "data/uk_host/{kmer_feat}feats_{split}.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.kmer_feat} host 10 {params.split} uk'

rule uk_host_average:
    input:
        expand("data/uk_host/{kmer_feat}feats_{split}.pkl", split = splits, kmer_feat = kmer_feats)
    output:
        "results/uk_host/host_{kmer_feat}feats_ANNtrainedOnuk_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} host uk'

rule uk_us_host_split:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "data/hyp_splits/uk_us-host/splits/set{split}/"
    shell:
        'python src/validation_split_hyperas.py uk_us host'

rule uk_us_host_hyperas:
    input:
        expand("data/hyp_splits/uk_us-host/splits/set{split}/", split = splits)
    output:
        "data/uk_us_host/{kmer_feat}feats_{split}.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.kmer_feat} host 10 {params.split} uk_us'

rule uk_us_host_average:
    input:
        expand("data/uk_us_host/{kmer_feat}feats_{split}.pkl", split = splits, kmer_feat = kmer_feats)
    output:
        "results/uk_us_host/host_{kmer_feat}feats_ANNtrainedOnuk_us_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} host uk_us'

rule us_host_split:
    input:
        'data/ukus_unfiltered/kmer_matrix.npy'
    output:
        "data/hyp_splits/us-host/splits/set{split}/"
    shell:
        'python src/validation_split_hyperas.py us host'

rule us_host_hyperas:
    input:
        expand("data/hyp_splits/us-host/splits/set{split}/", split = splits)
    output:
        "data/us_host/{kmer_feat}feats_{split}.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.kmer_feat} host 10 {params.split} us'

rule us_host_average:
    input:
        expand("data/us_host/{kmer_feat}feats_{split}.pkl", split = splits, kmer_feat = kmer_feats)
    output:
        "results/us_host/host_{kmer_feat}feats_ANNtrainedOnus_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}'
    shell:
        'python src/hyp_average.py {params.kmer_feat} host us'
