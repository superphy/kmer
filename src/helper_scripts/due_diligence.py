import os
import pandas as pd
import re
import math

log_list = []

path = os.path.abspath('data/biolog_csv/_07-2890_Escherichia coli__1_37_PMX_438_2#19#2019_A_ 1A_0.csv')
new_data = pd.read_csv(path)

hour = new_data.loc[:120, 'Hour':]
df0 = pd.DataFrame(hour)

#print(df0)

for row in df0["G10"]:
    log_list.append(math.log2(row))
sum = sum(log_list)
print(sum)
