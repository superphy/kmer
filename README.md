### kmer prediction

Using kmer counts to predict if an e. coli genome orginated from humans or bovines. 

### To Run

**Step One:** Run `python setup\_database.py`

**Step Two:** Since this is a work in progress update the file paths at the top of jellyfish\_test.py to point to the appropriate locations on your computer.

**Step Three:** Run `python jellyfish\_test.py` 


### Note

query\_kmer.py and kmer\_prediction.py can be ignored since they are from before the switch was made to jellyfish.

### Dependencies

- Scikit-learn
- lmdb
- numpy 
- jellyfish *You do not need to install the python wrapper for jellyfish, just the command line tool*
