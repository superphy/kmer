### kmer prediction

Using kmer counts as inputs to Support Vector Machines to predict if a genome is positive or negative for a phenotype.

Since jellyfish (the kmer counter used in this program) does not output kmers that have a count of zero and SVM's require that every input be of the same length the output from jellyfish cannot be directly input into an SVM. To overcome this problem only kmers that have a non-zero count in every genome are input into the svm.

ecoli_human_bovine.py, salmonella.py, and run.py are scripts for testing kmer_prediction.py on various inputs. kmer_prediction.py is the meat and potatoes of this program.

### To run kmer_prediction.py from the command line

`python kmer_prediction.py k l repetitions positive_path negative_path prediction_path`

- k: Length of kmer to use
- l: Minimum count required for a kmer to be output
- repetitions: Number of times to train and test the model. Ignored if prediction_path is not none.
- positive_path: Path to a directory containing fasta files for genomes positive for the phenotype.
- negative_path: Path to a directory containing fast files for genomes negative for the phenotype.
- prediction_path: None or Path to a directory contiaining fasta files for genomes that you would like to predict if they are positive or negative for a    phenotype. If None the file in positive_path and negative_path are shuffled and split into training and testing groups and the percentage that the model guesses correct is output. If prediction_path is not None a binary list is output, where a 1 means that the genome belongs to the positive_path group and 0 means that the genome belongs to the negative_path group.

### To use kmer_prediction in another script

`from kmer_prediction import run`
` output = run(k, l, repetitions, positive_path, negative_path, prediction_path)`

Where the parameters for run have the same specifications as the command line arguments when running kmer_prediction.py from the command line

### Dependencies

To install the dependecies run `conda create --name <env> --file conda-specs.txt`

You will need to install jellyfish separately. See https://github.com/gmarcais/Jellyfish for installation instructions. *You do not need to install the python binding for jellyfish, just the command line tool*

If you receive an import error stating that python can't find the module lmdb, run `pip install lmdb` with the conda environment activated.
