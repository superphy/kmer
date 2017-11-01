##Needs to be updated

### kmer prediction

Using kmer counts as inputs to machine learning models to predict if a genome is positive or negative for a phenotype.

Counts kmers using kmer_counter.py


### To use kmer_prediction.py

`from kmer_prediction import run`

`output = run(k, l, repetitions, positive_path, negative_path, prediction_path)`

- k: Length of kmer to use
- l: Minimum count required for a kmer to be output
- repetitions: Number of times to train and test the model. Ignored if prediction_path is not none.
- positive_path: Path to a directory containing fasta files for genomes positive for the phenotype.
- negative_path: Path to a directory containing fast files for genomes negative for the phenotype.
- prediction_path: None or Path to a directory contiaining fasta files for genomes that you would like to predict if they are positive or negative for a phenotype. If None the file in positive_path and negative_path are shuffled and split into training and testing groups and the percentage that the model guesses correct is output. If prediction_path is not None a binary list is output, where a 1 means that the genome belongs to the positive_path group and 0 means that the genome belongs to the negative_path group.


### kmer counter

Since jellyfish (the kmer counter used in this program) does not output kmers that have a count of zero and machine learning models require that every input be of the same length the output from jellyfish cannot be directly input into a ml model. To overcome this problem only kmers that have a non-zero count in every genome are stored in the database.

count kmers: Counts all kmers of length k that appear at least l times in each file in a list
of fasta files. Stores the output in a database

get_counts: Returns a list of the kmer counts stored in the database for each input fasta file.

add_counts: Adds new files to the database, does not affect kmer counts already in the database. Useful for when you train a model on a dataset and then later get more data. Since the kmer counts of the new data will have to have the same length as the kmer counts of the old data.


### To use kmer_counter.py

`from kmer_counter import count_kmers, add_kmers, get_counts`

`count_kmers(k, limit, files, database)`

`add_counts(new_files, database)`

`data = get_counts(files+new_files, database)`

- k: Length of kmer to use.
- limit: Minimum count required for a kmer to be output.
- files/new_files: Lists of paths to fasta files.
- database: Name of the lmdb database you would like to use.

- data: A list of lists of kmer counts, can be used as the input to a machine learning model.


### get_fasta_from_json.py

If you have pulled metadata from enterobase using Superphy/MoreSerotype/module/DownloadMetadata.py or something similar, this allows you to search the created json files and then use the corresponding fasta/fastq files on moria as input to a machine learning module. bad_genomes.csv and invalid_files.csv are used to determine that the files from enterobase are usable.

- get_fasta_from_json: Takes two json files returns lists of filepaths to the fasta files corresponding to the genomes whose metadata is in the json files
- train_test: Uses get_fasta_from_json, takes two json files returns the x_train, y_train, x_test, y_test corresponding to the genomes contained in the json files
- train_test_files: Writes the output of train_test to two files so that you can save the files you trained and tested on to see if your results are reproducible.


### US_UK_data.py

Sets up the files/data to attempt to replicate the results of the paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5056084/. human_bovine.csv is the metadata sheet for this data.

- get_filepaths: Returns x_train, y_train, x_test, y_test where x_train and x_test are paths to the fasta files and y_train and y_test are the human/bovine labels.

- get_preprocessed_data: Returns x_trian, y_train, x_test, y_test where x_train adn x_test are already preprocessed data ready to be immediately input into a machine learning model.


### salmonella_amr.py

Sets up files to to train a ml model on salmonella antimicrobial resistance.


### Dependencies

To install the dependecies run `conda create --name <env> --file conda-specs.txt`

You will need to install jellyfish separately. See https://github.com/gmarcais/Jellyfish for installation instructions. *You do not need to install the python binding for jellyfish, just the command line tool*

If you receive an import error stating that python can't find the module lmdb, run `pip install lmdb` with the conda environment activated.
