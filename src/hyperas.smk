attributes = ["Host"]
splits = ["1","2","3","4","5"]
datasets = ["kmer"]
feats=[i for i in range(100,3000,100)]+[i for i in range(3500,10500,500)]
rule all:
    input:
        expand("results/{dataset}_{attribute}/{attribute}_{feat}feats_ANNtrainedOn{dataset}_testedOnaCrossValidation.pkl", attribute = attributes, dataset = datasets, feat = feats)

rule split:
    input:
        expand("data/filtered/{attribute}/{dataset}_matrix.npy", attribute = attributes, dataset = datasets)
    output:
        "data/hyp_splits/{dataset}-{attribute}/splits/set1/"
    shell:
        'python src/validation_split_hyperas.py kmer {attributes}'

rule hyperas:
    input:
        expand("data/hyp_splits/{dataset}-{attribute}/splits/set1/",dataset = datasets, attribute = attributes)
    output:
        "data/{dataset}_{attribute}/{feat}feats_{split}.pkl"
    params:
        feat = '{feat}',
        attribute = '{attribute}',
        split = '{split}'
    shell:
        'python src/hyp.py {params.feat} {params.attribute} 10 {params.split} kmer'

rule average:
    input:
        expand("data/{dataset}_{attribute}/{feat}feats_{split}.pkl",dataset = datasets, attribute = attributes, split = splits, feat = feats)
    output:
        "results/{dataset}_{attribute}/{attribute}_{feat}feats_ANNtrainedOn{dataset}_testedOnaCrossValidation.pkl"
    params:
        feat = '{feat}',
        attribute = '{attribute}',
        dataset = '{dataset}'
    shell:
        'python src/hyp_average.py {params.feat} {params.attribute} {params.dataset}'
