from kpal.klib import Profile
import h5py
import numpy as np
import os
from sklearn import svm
from random import randrange


human_file_path = '/home/james/Human/'
bovine_file_path = '/home/james/Bovine/'

def convert_fasta_to_kmer_count_array(file, k):
    max_kmers = 4**k
    p = Profile.from_fasta(open(file), length=k, name='kmer')

    #Saves the kPAL 'Profile as a hdf5 file'
    p.save(h5py.File("temp.hdf5", "w"), 'kmer')

    #Open the saved file using h5py
    kmer_counts = h5py.File("temp.hdf5", "r")

    #Access the h5py group 'profiles'
    grp = kmer_counts['profiles']
    dataset = grp.require_dataset('kmer', shape=(max_kmers,), dtype='int64', exact=False)

    #Create np array to manipulate the dataset
    arr = np.zeros((max_kmers,), dtype='int64')

    #read the hdf5 file into the array
    dataset.read_direct(arr, np.s_[0:max_kmers], np.s_[0:max_kmers])

    return arr

def generate_training_data(begin, end, k):
    human_files = os.listdir(human_file_path)
    human_files = [human_file_path + x for x in human_files]
    bovine_files = os.listdir(bovine_file_path)
    bovine_files = [bovine_file_path + x for x in bovine_files]
    human_training = []
    bovine_training = []
    training_data = []
    training_answers = []
    size = end-begin+1
    for i in range(size):
        print "Creating training files #%d" % i
        human_training.append(convert_fasta_to_kmer_count_array(human_files[i+begin], k))
        bovine_training.append(convert_fasta_to_kmer_count_array(bovine_files[i+begin], k))
    h_count = 0;
    b_count = 0;
    while h_count < size and b_count < size:
        rand = randrange(0,10)
        if rand%2 == 0:
            training_data.append(human_training[h_count])
            training_answers.append("Human")
            h_count += 1
        else:
            training_data.append(bovine_training[b_count])
            training_answers.append("Bovine")
            b_count += 1
    while h_count < size:
        training_data.append(human_training[h_count])
        training_answers.append("Human")
        h_count += 1
    while b_count < size:
        training_data.append(bovine_training[b_count])
        training_answers.append("Bovine")
        b_count += 1

    return training_data, training_answers


def main():
    print "Generating Training Data...."
    X,Y = generate_training_data(0, 33, 10)
    print "Creating Support Vector Machine...."
    machine = svm.SVC()
    print "Training Support Vector Machine..."
    machine.fit(X,Y)
    print "Generating Test Data...."
    Z,ZPrime = generate_training_data(34, 43, 10)
    print "Creating Predictions...."
    output = machine.predict(Z)

    count = 0
    for i in range(len(Z)):
        if ZPrime[i] == output[i]:
            count += 1

    ans = (count*100)/len(Z)
    print "Percent Correct: %d" % ans



if __name__ == "__main__":
    main()
