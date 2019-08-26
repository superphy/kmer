import pandas as pd
from scipy.spatial.distance import squareform, pdist
from skbio.tree import nj

temp = []

with open("data/omnilog_data_summary.txt", 'r') as read:
    for line in read:
        temp.append(line.split('\t'))

for list in temp:
    num = list[2].split('\n')
    num = float(num[0])
    list[2] = num
    list.append(num)

df = pd.DataFrame(data = temp, columns = ["Strain", "Substrate", "AUC", "AUC"])
print(df.iloc[:, 2:])

distance = pd.DataFrame(squareform(pdist(df.iloc[:, 2:])), columns=df['Strain'], index=df['Strain'])

tree = nj(distance)
