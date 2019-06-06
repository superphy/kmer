import os
import pandas as pd


for filename in os.listdir('data/omnilog_data'):
    path = os.path.abspath('data/omnilog_data/{}'.format(filename))

    with open(path) as file:
        for line in file:
            if "Plate Type" in line:
                index = line.split(',')
                pm = index[1]
                pm.split()
                plate = pm[2]
                os.rename(path, 'data/PM{0}/{1}'.format(plate, filename))
