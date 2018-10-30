# Installation

1. Clone repository
1. Install dependecies: run `conda env create -f environment.yml`
1. Start conda environment: run `source activate <env>`
1. [Install jellyfish](https://github.com/gmarcais/Jellyfish "Jellyfish GitHub") __*You do not need to install the python binding for jellyfish, just the command line tool*__
1. Install kmerprediction: `python setup.py install`
1. Verify that everything is working: run `nose2` this will run all of the tests, they should all pass.

#### Some Common Errors:

* File/filepath does not exist
  * Update the filepaths in `kmerprediction/constants.py` to point to the correct locations on your machine
  * You will have to rerun `python setup.py install` in order for the changes to take effect.
* Import error stating that python can't find the module lmdb
  * run `pip install lmdb` with the conda environment activated.
* Error like:
    ```
    File "/home/user/miniconda3/envs/kmer/lib/python2.7/site-packages/hyperopt/pyll/base.py", line 715, in toposort
       assert order[-1] == expr
    TypeError: 'generator' object has no attribute '__getitem__'
    ```
  * Downgrade networkx from version 2.0 to version 1.1.


# Run Analysis from Paper

Update `kmerprediction/constants.py` so that `OMNILOG_FASTA`, `OMNILOG_DATA`,`GENOME_REGION_TABLE`, and `ECOLI` point to the correct locations on your machine. You will have to rerun `python setup.py install` in order for the changes to take effect. 

* `OMNILOG_FASTA`: Path to directory containing the Omnilog fasta files,
* `OMNILOG_DATA`: Path to the `wide_format_header.txt` file containing the Omnilog AUC data.
* `GENOME_REGION_TABLE`: Path to the binary table containing genome region presence/absence data for the US/UK _e. Coli_ samples.
* `ECOLI`: Path to the directory containing the US/UK _e. Coli_ fasta files.

## US/UK Analysis

Run `snakemake -s validation.smk`

## Omnilog Analysis

Run `snakemake -s omnilog.smk`

# kmerprediction information

See `kmerprediction/README.md`
