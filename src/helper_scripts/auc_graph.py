import pandas as pd

list = []
line_list = []

with open("data/omnilog_data_summary.txt", 'r') as read:
    for line in read:
        line_list.append(line)

for line in line_list:
    sec = line
    try:
        gene, substrate, auc = line.split(' ')
        temp_list = [gene, substrate, auc]
        list.append(temp_list)
    except:
        continue

print(list)
