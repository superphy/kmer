#!/usr/bin/env python

import numpy as np
import pandas as pd
import itertools

miss_counter = 0
s13k = np.load('split1_3000feats.npy')
s23k = np.load('split2_3000feats.npy')
s33k = np.load('split3_3000feats.npy')
s43k = np.load('split4_3000feats.npy')
s53k = np.load('split5_3000feats.npy')

for i in range(1,6):
     sx2k = np.load('split'+str(i)+'_2000feats.npy')
     for j in sx2k:
         if j not in itertools.chain(s13k,s23k,s33k,s43k,s53k):
             print(j)
