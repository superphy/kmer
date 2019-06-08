#!/usr/bin/env python

"""
This script turns omnilog_rows as genome names and converts it
to an array of serotypes and hosts for the ML models.
Needs omnilog_metadata.csv
"""

import numpy as np
import pandas as pd

def strain_to_y(strain, y):
    """
    takes in a strain name and what you want returned, e.g. Serotype or Host.
    For example, (ECI-2895, host) will return water
    """
    return metadata.loc[metadata['Strain'] == strain][y].values[0]

kmer_rows = np.load('data/unfiltered/omnilog_rows.npy')
#kmer_rows = [i.decode('utf-8') for i in kmer_rows]

metadata = pd.read_csv('data/omnilog_metadata.csv')

for col in ['Host', 'Serotype', 'Otype', 'Htype']:
    kmer_y = [strain_to_y(i,col) for i in kmer_rows]
    np.save("data/unfiltered/omnilog_rows_"+col, kmer_y)
