import numpy as np

arr = np.load("annotation/15mer_data/kmer_rows_Host.npy", allow_pickle = True)
#matrix = np.load("data/filtered/host/omnilog_matrix.npy", allow_pickle = True)

all_host = np.asarray(arr)

print(all_host)

for i in range(len(all_host)):
    if all_host[i] == "Human":
        all_host[i] = 1
    else:
        all_host[i] = 0

np.save("annotation/15mer_data/kmer_rows_Host2.npy", all_host)
print(all_host)
