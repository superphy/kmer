#!/usr/bin/env python

"""
Removes any classes below a certain threshold
"""
import numpy as np
import pandas as pd
import os
import collections
import sys

if __name__ == "__main__":
	source = sys.argv[1]

	min_thresh = 5

	if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/filtered/'):
		os.mkdir(os.path.abspath(os.path.curdir)+'/data/filtered/')

	matrix = np.load("data/unfiltered/"+source+"_matrix.npy")
	rows_genomes = np.load("data/unfiltered/"+source+"_rows.npy")
	Host = np.load("data/unfiltered/"+source+"_rows_Host.npy")
	Serotype = np.load("data/unfiltered/"+source+"_rows_Serotype.npy")
	Otype = np.load("data/unfiltered/"+source+"_rows_Otype.npy")
	Htype = np.load("data/unfiltered/"+source+"_rows_Htype.npy")

	for i, name in [[Host,'Host'], [Serotype,'Serotype'], [Otype,'Otype'], [Htype,'Htype']]:
		counts = collections.Counter(i)
		print(counts)
		counts = dict(counts)
		#classes = (list(counts.keys()))
		class_mask = np.array((len(i)),dtype = 'object')
		class_mask = np.asarray([counts[j]>=5 for j in i])

		if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/filtered/'+name):
			os.mkdir(os.path.abspath(os.path.curdir)+'/data/filtered/'+name)
		np.save('data/filtered/'+name+'/'+source+'_matrix.npy', matrix[class_mask])
		np.save('data/filtered/'+name+'/'+source+'_rows.npy', rows_genomes[class_mask])
		np.save('data/filtered/'+name+'/'+source+'_rows_'+name+'.npy', i[class_mask])
