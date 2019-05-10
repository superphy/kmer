import pandas as pd
import os

data = pd.read_excel('data/ectyper_output_20190211.xlsx')

df = data[['Name', 'O-type', 'H-type', 'LabWare serotype']].copy()
df['Host'] = 'Human'
df['WGS'] = '1'
df['LSPA6'] = 'N/A'
df.columns = ["Strain","Otype", "Htype", "Serotype",'Host', 'WGS', 'LSPA6']
df.to_csv('data/omni_test.csv', index=False)
#print(df)

df2 = pd.read_csv('data/omnilog_metadata.csv')

df3 = pd.concat([df, df2])
print(df3)
