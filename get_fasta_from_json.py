import random
import json



def check_fasta(filepath):
    """
    Takes a path to a file, returns True if the file is fasta or fastq, returns
    False otherwise.
    """
    with open(filepath, 'r') as f:
        firstline = f.readline()
        if firstline[0] == '>' or firstline[0] == '@':
            return True
        else:
            return False



def get_fasta_from_json(positive_json, negative_json):
    """
    Parmeters:
        positive_josn:  Path to a json file that contains a list of metadata
                        for enterobase genomes, where each genome possesses a
                        phenotype of interest.
        negative_json:  Path to a json file that contains a list of metadata
                        for enterobase genomes, where each genome does not
                        possess the phenotype of interest.
    Returns:
        Two lists (of approximately equal length) of complete file paths to
        valid fasta files from enterobase that are stored on moria. The first
        list contains only genomes positive for the phenotype of interest and
        the the second list contains only genomes that are negative for the
        phenotype of interest.

    See Superphy/MoreSerotype/module/DownloadMetadata.py on Github for a script
    that can generate the json files required to make this work.

    Since some of the genomes pulled down from enterobase are invalid, not all
    of the genomes contained in the json files will be output.
    """
    moria = '/home/rboothman/moria/enterobase_db/'
    bad = 'bad_genomes.csv'
    invalid = 'invalid_files.csv'

    with open(bad, 'r') as f:
        lines = f.read()
        bad_genomes = lines.split('\n')

    with open(invalid, 'r') as f:
        lines = f.read()
        invalid_genomes = lines.split('\n')

    with open(negative_json, 'r') as f:
        data = json.load(f)
        fasta_names = [str(x['assembly_barcode']) for x in data]
        negative_fasta = [moria+x+'.fasta' for x in fasta_names
                          if x not in bad_genomes
                          and moria+'x'+'.fasta' not in invalid_genomes
                          and check_fasta(moria+x+'.fasta')]

    with open(positive_json, 'r') as f:
        data = json.load(f)
        fasta_names = [str(x['assembly_barcode']) for x in data]
        positive_fasta = [moria+x+'.fasta' for x in fasta_names
                          if x not in bad_genomes
                          and moria+'x'+'.fasta' not in invalid_genomes
                          and check_fasta(moria+x+'.fasta')]

    return positive_fasta, negative_fasta



def train_test(positive_json, negative_json):
    """
    Parameters:
        See get_fasta_from_json()
    Returns:
        x_train:    A list of filepaths to genomes to be used to train a model.
        y_train:    A list of labels for each genome in x_train.
        x_test:     A list fo filepaths to genomes to be used to test a model.
        y_test:     A list of labels for each genome in x_test.

    Shuffles the output lists of get_fasta_from_json together to create an 80/20
    train/test split of the genomes as well lists of the corresponding labels. 1
    for possesing the phenotype of interest and 0 for not possessing the
    phenotype.

    See Superphy/MoreSerotype/module/DownloadMetadata.py on Github for a script
    that can generate the json files required to make this work.
    """
    pos_fasta, neg_fasta = get_fasta_from_json(positive_json, negative_json)
    labels = [1 for x in pos_fasta] + [0 for x in neg_fasta]
    fasta_files = pos_fasta + neg_fasta

    temp = list(zip(fasta_files, labels))
    random.shuffle(temp)
    fasta_files, labels = zip(*temp)

    fasta_files = list(fasta_files)
    labels = list(labels)

    cutoff = int(0.8*len(fasta_files))

    x_train = fasta_files[:cutoff]
    y_train = labels[:cutoff]
    x_test = fasta_files[cutoff:]
    y_test = labels[cutoff:]

    return x_train, y_train, x_test, y_test



def train_test_files(train_file, test_file, positive_json, negative_json):
    """
    Parameters:
        train_file:     Path to a file where you would like to store the paths
                        to the genomes that will be used to train a model.
        test_file:      Path to a file where you woule like to store the paths
                        to the genomes that will be used to test a model.
        psotive/negative_json:  See get_fasta_from_json()
    Returns:
        Nothing.

    Writes the output of train_test to "train_file" and "test_file", formats the
    files like csv files where the first column contains the path to the fasta
    file for a genome and the second column contains the label for the genome.

     See Superphy/MoreSerotype/module/DownloadMetadata.py on Github for a script
     that can generate the json files required to make this work.
    """
    x_train, y_train, x_test, y_test = train_test(positive_json, negative_json)

    with open(train_file, 'w') as f:
        for i in range(len(x_train)):
            output = "%s,%s\n" % (x_train[i], y_train[i])
            f.write(output)

    with open(test_file, 'w') as f:
        for i in range(len(x_test)):
            output = "%s,%s\n" % (x_test[i], y_test[i])
            f.write(output)
