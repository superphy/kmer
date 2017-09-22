### kmer prediction

Using kmer counts to predict if an e. coli genome orginated from humans or bovines.

Since jellyfish (the kmer counter used in this program) does not output kmers that have a count of zero and SVM's require that every input be of the same length the output from jellyfish cannot be directly input into an SVM. TO overcome this problem two methods have been implemented here. The first removes any kmer with a zero count in one genome from every genome, this method is implemented in jellyfish\_remove\_zeros.py. The second method adds in all the missing kmers with a count of zero to each genome, this method is implemented in jellyfish\_add\_zeros.py.  

### To run jellyfish\_add\_zeros.py from the command line

**Step One:** Decide on the length of kmer you want to use. Call it k

**Step Two:** Run `python setup_database.py k ` *If you want to use multiple lengths they can all be passed in at once*

**Step Three:** Since this is a work in progress update the file paths at the top of jellyfish\_add\_zeros.py to point to the appropriate locations on your computer.

**Step Four:** Run `python jellyfish_add_zeros.py k`

### To run jellyfish\_remove\_zeros.py from the command line

**Step One:** Decide on the length of kmer you want to use. Call it k

**Step Two:** Since this is a work in progress update the file paths at the top of jellyfish\_add\_zeros.py to point to the appropriate locations on your computer.

**Step Three:** Run `python jellyfish_remove_zeros.py k`

### To use jellyfish\_remove\_zeros.py jellyfish\_add\_zeros.py or setup\_database.py in another script

`from jellyfish_remove_zeros import run as remove_zeros`

`from jellyfish_add_zeros import run as add_zeros`

`from setup_database import run as set_db`

### Note 1

The setup\_database.py script should only need to be run once when you first want to use a k-value after that you should just need to run the jellyfish\_add\_zeros.py script, but if something goes wrong while a script is accessing a database, the databse may need to be reset, at which point setup\_database.py will need to be run again.

### Note 2

query\_kmer.py and kmer\_prediction.py can be ignored since they are from before the switch was made to jellyfish.

### Dependencies

- Scikit-learn
- lmdb
- numpy
- jellyfish *You do not need to install the python wrapper for jellyfish, just the command line tool*
