import numpy as np

arr = np.load("data/filtered/Serotype/omnilog_rows_Serotype.npy", allow_pickle = True)
#matrix = np.load("data/filtered/Serotype/omnilog_matrix.npy", allow_pickle = True)

all_serotype = np.asarray(arr)

print(all_serotype)

for i in range(len(all_serotype)):
    if all_serotype[i] == "O157:H7":
        all_serotype[i] = 1
    else:
        all_serotype[i] = 0

np.save("data/filtered/Serotype/omnilog_rows_Serotype2.npy", all_serotype)
print(all_serotype)
