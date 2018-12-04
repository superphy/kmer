## Setup
1. Clone repository (run `git clone https://github.com/superphy/kmer.git`)
2. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
3. Install dependecies: run `conda env create -f data/envi.yaml`
4. Start conda environment: run source activate skmer
5. Run the following command to prepare the data, where 'X' is the number of cores you wish to use

   `snakemake -j X -s src/setup.smk`
6. Run the following command to run all of the tests

   `snakemake -s src/run_tests.smk`
   
7. Run `for dir in results/*; do python src/figures.py $dir/; done` to save all the results as figures


### Manually Running Tests
If you do not want to run all tests in step 6 of setup, run `src/model.py --help` to see how to run the model for a specific set of parameters.
