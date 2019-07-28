import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from operator import itemgetter


df = pd.read_csv("data/final_omnilog_metadata.csv")

Otype = df.pivot_table(index=['Otype'], aggfunc='size')
Htype = df.pivot_table(index=['Htype'], aggfunc='size')
Host = df.pivot_table(index=['Host'], aggfunc='size')
Serotype = df.pivot_table(index=['Serotype'], aggfunc='size')

Otype = dict(Otype)
Htype = dict(Htype)
Host = dict(Host)
Serotype = dict(Serotype)


for x, i in enumerate([Otype, Htype, Host, Serotype]):
    print(x)
    dictlist = []
    for key, value in i.items():
        temp = [key,value]
        dictlist.append(temp)
    dictlist = sorted(dictlist, key=itemgetter(1))
    isolate_df = pd.DataFrame(data = dictlist, columns = ["Type", "Number of Samples"])
    #print(isolate_df)

    graph = sns.barplot(x = "Type", y = "Number of Samples", data = isolate_df)

    if x == 0:
        plt.title("Otype")
        plt.xticks(rotation=90, fontsize=10)
        plt.tight_layout(pad = 1)
        plt.savefig('figures/Otype.png')
        plt.clf()
    elif x == 1:
        plt.title("Htype")
        plt.xticks(rotation=90, fontsize=10)
        plt.tight_layout(pad = 1)
        plt.savefig('figures/Htype.png')
        plt.clf()
    elif x == 2:
        plt.title("Host")
        plt.xticks(rotation=90, fontsize=10)
        plt.tight_layout(pad = 1)
        plt.savefig('figures/Host.png')
        plt.clf()
    else:
        plt.title("Serotype")
        plt.xticks(rotation=90, fontsize=10)
        plt.tight_layout(pad = 1)
        plt.savefig('figures/Serotype.png')
        plt.clf()
