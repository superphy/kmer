attributes = ["Host"]
splits = ["1","2","3","4","5"]
dataset = ["kmer"]
feats=[i for i in range(100,3000,100)]+[i for i in range(3000, 10500, 500)]
rule all:
    input:
        expand("results/{dataset}_{attributes}/{attributes}_{feat}feats_ANNtrainedOn{dataset}_testedOnaCrossValidation.pkl", attributes = attributes, feat = feats, dataset = dataset)

rule split:
    output:
        "data/filtered/{attributes}/splits/set1/"
    params:
        attributes  = "{attributes}"
    shell:
        "sbatch -c 1 --mem 32G --wrap='python src/validation_split_hyperas.py kmer {params.attributes}'"

rule hyperas:
    input:
        expand("data/filtered/{attributes}/splits/set1/", attributes = attributes)
    output:
        "data/{attributes}/hyperas/{feat}feats_{split}.pkl"
    params:
        attributes = "{attributes}",
        split = "{split}",
        feat = "{feat}"
    shell:
        "sbatch -c 16 --mem 125G --wrap='python src/hyp.py {params.feat} {params.attributes} 10 {params.split} {params.dataset}'"

rule average:
    input:
        expand("data/{attributes}/hyperas/{feat}feats_{split}.pkl", attributes = attributes, feat = feats, split = splits)
    output:
        "results/{params.dataset}_{attributes}/{attributes}_{feat}feats_ANNtrainedOn{params.dataset}_testedOnaCrossValidation.pkl"
    params:
        attributes = "{attributes}",
        feat = "{feat}"
    shell:
        "sbatch -c 1 --mem 2GB --wrap='python src/hyp_average.py {params.feat} {params.attributes} {params.dataset}'"
