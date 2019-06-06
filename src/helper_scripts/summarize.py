import pandas as pd
import os


results_list = []
for filename in os.listdir('results/csv/summary'):
    path = os.path.abspath('results/csv/summary/{}'.format(filename))
    with open(path) as file:
        data = pd.read_csv(file)
        for index, row in data.iterrows():
            if "ANN" in row["model"] and row['feats'] == 1000 or row['feats'] == 190:
                acc, model, train, test, feats, attribute = row['acc'], row['model'], row['train'], row['test'], row['feats'], row['attribute']
                temp_list = []
                temp_list = acc*100, model, train, test, feats, attribute
                results_list.append(temp_list)
            if "SVM" in row["model"] and row['feats'] == 3000 or row['feats'] == 190:
                acc, model, train, test, feats, attribute = row['acc'], row['model'], row['train'], row['test'], row['feats'], row['attribute']
                temp_list = []
                temp_list = acc*100, model, train, test, feats, attribute
                results_list.append(temp_list)
            if "XGB" in row["model"] and row['feats'] == 3000 or row['feats'] == 190:
                acc, model, train, test, feats, attribute = row['acc'], row['model'], row['train'], row['test'], row['feats'], row['attribute']
                temp_list = []
                temp_list = acc*100, model, train, test, feats, attribute
                results_list.append(temp_list)

results_df = pd.DataFrame(data = results_list, columns = ["Accuracy", "Model", "Train", "Test", "Number of Features", "Attribute"])
print(results_df)
results_df.to_csv('results/csv/results.csv', index = False)
