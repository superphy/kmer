attributes = ["Host", "Serotype", "Otype", "Htype"]
models = ["XGB"]
splits = ["1","2","3","4","5"]
kmer_feats = [i for i in range(100,3000,100)]
omnilog_feats = [i for i in range(10,190,10)]
omnilog_dataset = "omnilog"
kmer_dataset = "kmer"

rule all:
    input:
        expand("results/kmer_{attribute}/{attribute}_{kmer_feat}feats_{model}trainedOnkmer_testedOnaCrossValidation.pkl", attribute = attributes, kmer_feat = kmer_feats, model = models),
        expand("results/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_{model}trainedOnomnilog_testedOnaCrossValidation.pkl", attribute = attributes, omnilog_feat = omnilog_feats, model = models),
        expand("results/uk_host/Host_{kmer_feat}feats_{model}trainedOnuk_testedOnaCrossValidation.pkl", kmer_feat = kmer_feats, model = models),
        expand("results/uk2us_host/Host_{kmer_feat}feats_{model}trainedOnuk_testedOnus.pkl", kmer_feat = kmer_feats, model = models),
        expand("results/ukus_host/Host_{kmer_feat}feats_{model}trainedOnukus_testedOnaCrossValidation.pkl", kmer_feat = kmer_feats, model = models),
        expand("results/ukus2kmer_host/Host_{kmer_feat}feats_{model}trainedOnukus_testedOnkmer.pkl", kmer_feat = kmer_feats, model = models),
        expand("results/us_host/Host_{kmer_feat}feats_{model}trainedOnus_testedOnaCrossValidation.pkl", kmer_feat = kmer_feats, model = models),
        expand("results/us2uk_host/Host_{kmer_feat}feats_{model}trainedOnus_testedOnuk.pkl", kmer_feat = kmer_feats, model = models),
        expand("results/kmer2ukus_host/Host_{kmer_feat}feats_{model}trainedOnkmer_testedOnukus.pkl", kmer_feat = kmer_feats, model = models)
    #run:
        #shell("snakemake -s src/hyperas.smk")
        #shell("snakemake -s src/ukus_hyperas.smk")

rule kmer:
    input:
        expand("data/filtered/{attribute}/kmer_matrix.npy", attribute = attributes)
    output:
        "results/kmer_{attribute}/{attribute}_{kmer_feat}feats_{model}trainedOnkmer_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}',
        attribute = '{attribute}'
    shell:
        'python src/model.py -i -x kmer -a {params.attribute} -o results/kmer_{params.attribute} -f {params.kmer_feat} -m {params.model}'

rule omnilog:
    input:
        expand("data/filtered/{attribute}/omnilog_matrix.npy", attribute = attributes)
    output:
        "results/omnilog_{attribute}/{attribute}_{omnilog_feat}feats_{model}trainedOnomnilog_testedOnaCrossValidation.pkl"
    params:
        omnilog_feat = '{omnilog_feat}',
        model = '{model}',
        attribute = '{attribute}'
    shell:
        'python src/model.py -i -x omnilog -a {params.attribute} -o results/omnilog_{params.attribute} -f {params.omnilog_feat} -m {params.model}'

rule uk_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/uk_host/Host_{kmer_feat}feats_{model}trainedOnuk_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x uk -a Host -o results/uk_host -f {params.kmer_feat} -m {params.model}'

rule us_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/us_host/Host_{kmer_feat}feats_{model}trainedOnus_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x us -a Host -o results/us_host -f {params.kmer_feat} -m {params.model}'

rule ukus_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/ukus_host/Host_{kmer_feat}feats_{model}trainedOnukus_testedOnaCrossValidation.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x uk_us -a Host -o results/ukus_host -f {params.kmer_feat} -m {params.model}'

rule us2uk_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/us2uk_host/Host_{kmer_feat}feats_{model}trainedOnus_testedOnuk.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x us -y uk -a Host -o results/us2uk_host -f {params.kmer_feat} -m {params.model}'

rule uk2us_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/uk2us_host/Host_{kmer_feat}feats_{model}trainedOnuk_testedOnus.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x uk -y us -a Host -o results/uk2us_host -f {params.kmer_feat} -m {params.model}'

rule kmer2ukus_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/kmer2ukus_host/Host_{kmer_feat}feats_{model}trainedOnkmer_testedOnukus.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x kmer -y uk_us -a Host -o results/kmer2ukus_host -f {params.kmer_feat} -m {params.model}'

rule ukus2kmer_host:
    input:
        'data/uk_us_unfiltered/kmer_matrix.npy'
    output:
        "results/ukus2kmer_host/Host_{kmer_feat}feats_{model}trainedOnukus_testedOnkmer.pkl"
    params:
        kmer_feat = '{kmer_feat}',
        model = '{model}'
    shell:
        'python src/model.py -i -x uk_us -y kmer -a Host -o results/ukus2kmer_host -f {params.kmer_feat} -m {params.model}'
