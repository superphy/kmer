# Installation

1. Clone repository
1. Install dependecies: run `conda create --name <env> --file conda-specs.txt`
1. Start conda environment: run `source activate <env>`
1. [Install jellyfish](https://github.com/gmarcais/Jellyfish "Jellyfish GitHub") __*You do not need to install the python binding for jellyfish, just the command line tool*__
1. Verify that everything is working: run `nose2` this will run all of the tests, they should all pass.

#### Some Common Errors:

* File/filepath does not exist
  * Update the filepaths in constants.py to point to the correct locations on your machine
* Import error stating that python can't find the module lmdb
  * run `pip install lmdb` with the conda environment activated.
* Error like:
    ```
    File "/home/user/miniconda3/envs/kmer/lib/python2.7/site-packages/hyperopt/pyll/base.py", line 715, in toposort
       assert order[-1] == expr
    TypeError: 'generator' object has no attribute '__getitem__'
    ```
  * Downgrade networkx from version 2.0 to version 1.1.


# run.py
This is essentially the main method.

Provides a wrapper to gather data, preprocess the data, perform feature selection, build the model, train the model, and test/use the model. Chains together methods from get_data.py, feature_selection.py, feature_scaling, data_augmentation.py, and models.py.

#### To Use From the Command Line:

```
python run.py -i [config file] -o [output file] -n [name of run to use in outputfile]
```

```
usage: run.py [-h] [-i INPUT] [-o OUTPUT] [-n NAME]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        yaml configuration file for run. If not provided
                        Data/config.yml is used.
  -o OUTPUT, --output OUTPUT
                        yaml file where the results of the run will be stored.
                        If not provided Data/run_results.yml is used.
  -n NAME, --name NAME  What the yaml document will be named in the output
                        file. If not provided the current Datetime is used. If
                        using spaces surround with quotes.
```


The input file should be a yaml file specifying all of the arguments to use during the run. An example can be found in Data/config.yaml

Running the script multiple times with the same output file will not overwrite the previous results. Each time a run is performed a new yaml document is created and appended to the bottom of the output file. Each yaml document will be a dictionary with two keys 'name' and 'output' where 'name' contains either the datatime of the run or the name provided by the user and 'output' contains a dictionary that holds the results from the run and all of the parameters specified in the input file.

#### To Use in Another Script:

To get the same behaviour as from the command line:

```python
from run import main
main('config_file.yml', 'output_file.yml', 'name of run')
```

Or to get a results dictionary with the default parameters:

```python
from run import run
output = run()
```

Or to get a results dictionary with custom parameters:

```python
from models import neural_network as nn
from get_data import get_genome_region_us_uk_split as data
from feature_selection import variance_threshold as sel
from run import run

output = run(model=nn, data_method=data, selection=sel,
             selection_args={'threshold': 0.01}, scaler=None, reps=1,
             validate=True)
```

It is also possible to skip run.py altogether and do something like:

```python
from models import neural_network
from get_data import get_genome_region_us_uk_mixed as data
from feature_selection import variance_threshold as sel

d = data()
d = sel(d[0], threshold=0.01)
score = neural_network(d)
```

*The above is necessary if you want to change the order in which things occur, for instance performing data augmentation before performing feature scaling.*


## get_data.py

A collection of methods that gather and prepare data to be input into a machine learning model.

Most return: ((x_train, y_train, x_test, y_test), feature_names, test_files, LabelEncoder) where:
* **x_train**: 2D array with shape (number of samples, number of features) containing the training data.
* **y_train**: 1D array with shape (number of samples,) containing the classification labels for the training data.
* **x_test**: 2D array of the shape (number of test samples, number of features) containing the test data.
* **y_test**: Either a 1D array of the shape (number of test samples,) containing the classification labels for the test data or in the case where you are not validating the model is an empty array.
* **feature_names**: List of all the features present in each sample.
* **test_files**: List of the names of each input being used to test the model.
* **LabelEncoder**: Scikit-learn LabelEncoder object that will allow you to convert the predicted classifications back into a human readable format.

Some of the methods simply return: (x_train, y_train, x_test, y_test)

To recreate the results of the the [Lupolova et. al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5056084/ "NCBI article") paper with kmer counts as inputs use the method: get_kmer_us_uk_split. To recreate the results of the paper using genome region presence absence data use the method get_genome_region_us_uk_split.

The methods that prepare kmer data use kmer_counter.py to count the kmers.


## feature_selection.py

A collection of methods that perform feature selection on the training and testing data.

Each method has two positional arguments input_data and feature_names. input_data should be a tuple containing (x_train, y_train, x_test, y_test) as defined under get_data.py. feature_names should be a list containing the names of all the features in each sample or None. If feature_names is given the features that are removed from input_data by the feature selection will also be removed from feature_names.

All of the methods return input_data and feature_names with features removed from x_train, x_test and feature_names based on the conditions specified by the method and by the parameters passed to the method.


## feature_scaling.py

A collection of methods that perform feature scaling.

Each of the methods has a positional argument, input_data, as defined under feature_selection.py and returns input_data with the values in x_train and x_test scaled according to the specifications of the method and it's given parameters.


## data_augmentation.py

A collection of methods that perform data augmentation, the process of artificially generating new samples based on the samples you already have.

Each method has a positional argument, input_data, as defined under feature_selection.py and returns input_data with additonal samples added to x_train and y_train. x_test and y_test are not changed.


## models.py

A collection of methods containing machine learning models.

Each method has a positional argument, input_data, as defined under feature_selection.py and a named argument, validate. Validate should be a bool. If validate is True, the method will return an accuracy score representing the percentage of samples in x_test that were corretly classified and y_test must be given. If validate is False, the method will return a list containing the predicted classification for each sample in x_test and y_test is ignored.


## kmer_counter.py

Methods to count kmers, store the counts in a database, and then retrieve the counts later. The [jellyfish](https://github.com/gmarcais/Jellyfish "Jellyfish GitHub") program is used to count the kmers.

Since each data sample input to a machine learning model must have the same features as every other sample passed to the model the output of jellyfish can not be directly input into a machine learning model as it is possible that a kmer will appear in some, but not all of the samples. Therefore kmer_counter.py removes all kmers that do not appear at least "limit" times in each input genome.

The three methods useful to a user in kmer_counter.py are:
* **count kmers**: Counts all kmers of length k that appear at least limit times in each given fasta file. Stores the output in a database.
* **get_counts**: Returns a list of the kmer counts stored in the database for each input fasta file.
* **add_counts**: Adds new files to the database, does not affect kmer counts already in the database. Useful for when you train a model on a dataset and then later get more data. Since the kmers present in the new data must match the kmers present in the old data.


#### To use in another script:

```python
from kmer_counter import count_kmers, add_kmers, get_counts
count_kmers(k, limit, files, database)
add_counts(new_files, database)
data = get_counts(files+new_files, database)
```

- k: Length of kmer to use.
- limit: Minimum count required for a kmer to be output.
- files/new_files: Lists of paths to fasta files.
- database: Name of the lmdb database you would like to use.
- data: A list of lists of kmer counts, can be used as the input to a machine learning model.
