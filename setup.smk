
rule all:
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/kmer.smk")
        shell("snakemake -j {threads} -s src/uk_us.smk")
        shell("python src/y_builder.py")
        shell("python src/y_uk_us.py")
        shell("python src/omnilog_matrix.py")
        shell("python src/y_omnilog.py")
        shell("python src/remove_low_freq.py omnilog")
        shell("python src/remove_low_freq.py kmer")
