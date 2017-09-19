import lmdb
from numpy import base_repr
import sys

def kmer_rckmer(number, k):
    number = base_repr(number, base=4, padding=k)
    number = number[-k:]

    kmer = ['A' if x == '0'
            else 'C' if x == '1'
            else 'G' if x == '2'
            else 'T' for x in list(number)]
    reverse = kmer[-1::-1]
    reverse = ['A' if x == 'T'
                else 'C' if x == 'G'
                else 'G' if x == 'C'
                else 'T' for x in reverse]

    return ''.join(kmer), ''.join(reverse)


def set_db(k):
    environment = lmdb.open('kmer_counts', map_size=int(2e9), max_dbs=12)
    database = environment.open_db('k%d'%k, dupsort=False)
    with environment.begin(write=True, db = database) as transaction:
        transaction.drop(database, delete=False)
        i = 0
        while i < 4**k:
            kmer, rc_kmer = kmer_rckmer(i, k)
            if kmer <= rc_kmer:
                transaction.put(kmer, '0')
            else:
                transaction.put(rc_kmer, '0')
            i+=1
    environment.close()

def main():
    for i in sys.argv:
        if i.isdigit():
            print "Setting up database for k = %d"%i
            set_db(i)

if __name__ == "__main__":
    main()
