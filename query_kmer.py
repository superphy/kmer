from kpal.klib import Profile
import h5py
import numpy as np
from math import log

def query_kmer(hdf5_file, *kmers):
    '''
    Queries a numpy array, to find the kmer count of a specific base pair
    sequence.

    Args:
        hdf5_file:  An HDF5 file containing the results of a kmer count
                    operation. Designed to be used with the kPAL and h5py python
                    libraries after something along the lines of:

                        p=Profile.from_fasta(open(example.fasta), 3, 'test')
                        p.save(h5py.File("example.hdf5", "w"), 'test')

                    Then the file example.hdf5 would be passed to query_kmer().

        *kmers:     One or many base pair sequences to search for in the query.
                    Each must have the same length (which must match the length
                    used to generate the kmers, in the above example 3) and only
                    include a,A,c,C,g,G,t,t

    Returns:
        A list of strings. One string corresonding to each input base pair
        sequence.
    '''

    kmer_counts = h5py.File(hdf5_file, 'r')
    grp = kmer_counts['profiles']

    grp_name = grp.keys()[0]
    kmer_array = grp.get(grp_name, [])

    #Since the number of entries in the input array is the number of possible
    #"mers" of length k we can determine k from the length of the input array
    kmer_size = int(log(len(kmer_array))/log(4))

    output = []
    for kmer in kmers:
        #Check that the provided kmer is the correct length
        if not len(kmer)==kmer_size:
            output.append("The kmer %s is the incorrect length" % kmer)
        else:
            kmer = kmer.upper()

            #Verify that the kmer is valid
            new = [x for x in list(kmer) if x in ['A','C','G','T']]
            if not len(new) == len(kmer):
                output.append("The kmer %s is not valid" % kmer)
            else:
                #Convert the kmer to a base 4 number
                base4 = [0 if x == 'A'
                    else 1 if x == 'C'
                    else 2 if x == 'G'
                    else 3 for x in list(kmer)]

                #Convert form base 4 to base 10 to find the index
                base10 = 0
                i = len(kmer)-1
                k = 0
                while i >= 0:
                    base10 += (base4[i]*(4**k))
                    k += 1
                    i -= 1

                #Display the query result
                output.append("%s Count: %d" % (kmer, kmer_array[base10]))

    return output
