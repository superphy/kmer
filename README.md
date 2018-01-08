## run.py

Wrapper to gather data, preprocess the data, perform feature selection, build the model, train the model, and test/use the model.

### Command Line Usage

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
                        using spaces, surround with quotes.
```


The input file should be a yaml file specifying all of the arguments to use during the run. An example can be found in Data/config.yaml

The output file will also be a yaml file containing the complete results from the run as well as all the parameters used in the run.

### To use in another script

To run with the default parameters:

```python
from run import run
output = run()
```

Example with custom parameters:

```python
from models import neural_network as nn
from get_data import get_genome_region_us_uk_mixed as data
from feature_selction import variance_threshold as sel
from run import run

output = run(model=nn, data=data, selction=sel, scaler=None, reps=1, validate=True)
```

It is also possible to forget about run.py and do something like this:

```python
from models import neural_network_validation
from get_data import get_genome_region_us_uk_mixed as data
from feature_selction import variance_threshold as sel

d = data()
d = sel(*d)
score = neural_network_validation(*d)
```

The above is necessary if you want to change the order in which things occurr, for instance performing data augmentation before performing feature scaling.


## data.py

A collection of methods that prepare data to be input into machine learning models. Most return x_train, y_train, x_test, and y_test. Where x_train is the training data, y_train is the corresponding labels, x_test is the testing data, and y_test is the corresponding labels.

To recreate the results of the the Lupolova et. al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5056084/) with kmer counts as inputs use the method: get_kmer_us_uk_split. To recreate the results of the paper using genome region presence absence data use the method get_genome_region_us_uk_split.

See the individual methods for their necessary parameters and usage.

You will need to update the global human_path and bovine_path variables at the top of the file.


## models.py

A collection of ready to use machine learning models. Some take x_train, y_train, x_test, y_test as input and return an accuracy score after training and testing on those inputs. Some take x_train, y_train, and x_test and return predictions on x_test.

See the individual methods for their necessary parameters and usage.


## data_augmentaion.py

A collection of methods to augment data. All of them take x_train, y_train, x_test, and y_test and return x_train, y_train, x_test, y_test with additonal, (artificially generated) samples added to x_train and y_train.

See the individual methods for their necessary parameters and usage.


## feature_selection.py

A collection of methods that perform feature selection on the training data. All of them take x_train, y_train, x_test, y_test, the (training data and labels, and the test data and labels) as well as some method specific parameters. If you do not have labels for the test data simply pass None in place of y_test.
All of the methods return x_train, y_train, x_test, and y_test with some features removed from x_trian and x_test.

See the individual methods for their necessary parameters and usage.


## feature_scaling.py

A collection of methods that perform feature scaling on input data. All of them take x_train, y_train, x_test, y_test and return x_train, y_train, x_test, y_test with the data in x_train and x_test scaled according to the method.

See the individual methods for their necessary parameters and usage.


## Extending the above files

If you wish to add methods to any of the above files and would like the methods to work with run.py the methods should follow these guidelines:

- A Data method should return x_train, y_train, x_test, y_test
- Any method that will manipulate the output of a data method should have as parameters x_train, y_train, x_test, y_test and then args, where args is a list of method specific positional arguments.
- Any method whose output could be passed to a model should return x_train, y_train, x_test, y_test
- The methods that contain the models should perform all of the building, compiling, and training necessary for the model.
- Once a method is added to a file, the dictionary inside the get_methods() function for that file should be updated to contain the new method.


## kmer_counter.py

Methods to count kmers, store the counts in a database, and then retrieve the counts later. The jellyfish program, https://github.com/gmarcais/Jellyfish, is used to count the kmers.

In order to be meaningful the inputs to machine learning models must all be of the same length and the features of each input must match up, therefore the output of jellyfish can not be directly input into a machine learning model, instead only kmers that appear at least "limit" times in each input genome will be kept.

count kmers: Counts all kmers of length k that appear at least limit times in each given fasta file. Stores the output in a database

get_counts: Returns a list of the kmer counts stored in the database for each input fasta file.

add_counts: Adds new files to the database, does not affect kmer counts already in the database. Useful for when you train a model on a dataset and then later get more data. Since the kmer counts of the new data must match up with the kmer counts of the old data.


### To use in another script

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


### To use from the command line

Not supported


## Dependencies

To install the dependecies run `conda create --name <env> --file conda-specs.txt`

You will need to install jellyfish separately. See https://github.com/gmarcais/Jellyfish for installation instructions. *You do not need to install the python binding for jellyfish, just the command line tool*

If you receive an import error stating that python can't find the module lmdb, run `pip install lmdb` with the conda environment activated.

If, when running a script that involves hyperas, you receive an error like:

```sh
File "/home/user/miniconda3/envs/kmer/lib/python2.7/site-packages/hyperopt/pyll/base.py", line 715, in toposort
   assert order[-1] == expr
TypeError: 'generator' object has no attribute '__getitem__'
```

Try downgrading networkx from version 2.0 to version 1.1 and rerunning the script.
