## run.py

Wrapper to gather data, pass it to a machine learning model and then return either an accuracy score or prediction values.

### To use in another script

```
from run import run
output = run(model, data, reps, validata, params)
```

- model: The machine learning model to be used, see best_models.py
- data: The method used to prepare the data for the model, see data.py
- reps: How many times to run the model, if doing validation
- params: A list or tuple of all parameters to be passed to "data"
- validate: If true "data" should return x_train, y_train, x_testand y_test and "model" should accept the output of dataand return an accuracy. If false "data" should returnx_train, y_train, and x_test and "model" should acceptthe output of "data" and return predictions for x_test.


- Returns: The output of "model" when given "data". If validating the model, the output is the average over all repetitions.


### To use from the command line

`python run.py --model <name of ml model> --data <name of data method> --reps <# of repetitions to run> --validate <True or False> --params <the parameters to pass to data>`

All of the command line options are optional, any that are ommitted will be replaced by their default values. All of the options have short forms based on their first letter, for example --model and -m are equivalent.


## kmer_counter.py

Methods to count kmers, store the counts in a database, and then retrieve the counts later. The jellyfish program, https://github.com/gmarcais/Jellyfish, is used to count the kmers.

In order to be meaningful the inputs to machine learning models must all be of the same length and the features of each input must match up, therefore the output of jellyfish can not be directly input into a machine learning model, instead only kmers that appear at least "limit" times in each input genome will be kept.

count kmers: Counts all kmers of length k that appear at least limit times in each given fasta file. Stores the output in a database

get_counts: Returns a list of the kmer counts stored in the database for each input fasta file.

add_counts: Adds new files to the database, does not affect kmer counts already in the database. Useful for when you train a model on a dataset and then later get more data. Since the kmer counts of the new data must match up with the kmer counts of the old data.


### To use in another script

```
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


## data.py

A collection of methods that prepare data to be input into machine learning models. Most return x_train, y_train, x_test, and y_test. Where x_train is the trianing data, y_train is the corresponding labels, x_test is the testing data, and y_test is the corresponding labels.

To recreate the results of the the Lupolova et. al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5056084/) with kmer counts as inputs use the method: get_kmer_us_uk_split. To recreate the results of the paper using genome region presence absence data use the method get_genome_region_us_uk_split.

See the individual methods for their necessary parameters and usage.


## best_models.py

A collection of ready to use machine learning models. Some take x_train, y_train, x_test, y_test as input and return an accuracy score after training and testing on those inputs. Some take x_train, y_train, and x_test and return predictions on x_test.

See the individual methods for their necessary parameters and usage.


## data_augmentaion.py

A collection of methods to augment data. All of them take x, y the training data and corresponding labels as well as some method specific parameters, all of them return x, y with additonal, (artificially generated) samples added to x and their labels added to y.

See the individual methods for their necessary parameters and usage.


## feature_selection.py

A collection of methods that perform feature selection on the training data. All of them take x_train, y_train, x_test, y_test, the (training data and labels, and the test data and labels) as well as some method specific parameters. If you do not have labels for the test data simply pass None in place of y_test.
All of the methods return x_train, y_train, x_test, and y_test with some features removed from x_trian and x_test.

See the individual methods for their necessary parameters and usage.


## Dependencies

To install the dependecies run `conda create --name <env> --file conda-specs.txt`

You will need to install jellyfish separately. See https://github.com/gmarcais/Jellyfish for installation instructions. *You do not need to install the python binding for jellyfish, just the command line tool*

If you receive an import error stating that python can't find the module lmdb, run `pip install lmdb` with the conda environment activated.

If, when running a script that involves hyperas, you receive an error like:

```
File "/home/user/miniconda3/envs/kmer/lib/python2.7/site-packages/hyperopt/pyll/base.py", line 715, in toposort
   assert order[-1] == expr
TypeError: 'generator' object has no attribute '__getitem__'
```

Try downgrading networkx from version 2.0 to version 1.1 and rerunning the script.
