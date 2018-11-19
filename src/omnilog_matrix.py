#!/usr/bin/env python

import numpy as np
import pandas as pd
from copy import deepcopy
import collections
import re
import sys
from os import listdir

def find_index(element, array):
	for i, ele in enumerate(array):
		if(ele == element):
			return i
	return -1

def pad_zeros(genomeid):
	if (genomeid[:3]=='ECI'):
		eci, num = genomeid.split('-')
		return'{0}-{1:0>4}'.format(eci, num)
	else:
		return genomeid

def find_eci(genomeid):
	#print(genomeid)
	return re.search(r'(ECI-\d{1,4})|(ON-2011)|(Sakai)|(EDL933)', genomeid).group()
	#[\(ECI\-\d{1,4}\)\(ON\-2011\)\(Sakai\)]

def intersection(lst1, lst2):
	temp = set(lst2)
	lst3 = [value for value in lst1 if value in temp]
	return lst3

if __name__ == "__main__":
	row_names = pd.read_csv('data/omnilog_metadata.csv')
	row_names = row_names.values[:,0]
	omni_df = pd.read_csv('data/omnilog_data_summary.txt', delimiter = '\t')
	#pm = pd.read_csv('data/omnilog_well_descriptions.txt', delimiter = '\t')

	omnilog_cols = np.empty(192,dtype = 'object')
	all_feats = np.unique(omni_df.values[:,1])
	print(all_feats)
	print(find_index('Sucrose', all_feats))
	all_feats_matrix = np.zeros((len(row_names),len(all_feats)), dtype='object')
	all_feats_matrix[:] = -1

	coutnt=0
	for i, row in enumerate(omni_df.values):
		row[0] = pad_zeros(row[0])
		rowindx = find_index(row[0], row_names)
		colindx = find_index(row[1], all_feats)
		#print(rowindx, colindx)
		if(rowindx != -1 and colindx != -1):
			all_feats_matrix[rowindx,colindx] = row[2]
			coutnt +=1
	print(coutnt)
	print('start:',all_feats_matrix)
	print('start:',all_feats_matrix.shape)

	"""
		Matrix currently has all possible features against all possible genomes,
		features from omnilog_data_summary, genomes from omnilog_metadata
	"""

	first = []
	PM1_2 = []
	for i in range(1,21):
		print('***** PM'+str(i)+' *****')
		PM1 = listdir('data/PM'+str(i))
		PM1 = [find_eci(i) for i in PM1]
		PM1 = [pad_zeros(i) for i in PM1]
		if(i==1):
			first = PM1
		print(sorted(PM1))
		if(i==2):
			print('***** INTERSECTION *****')
			PM1_2 = intersection(PM1, first)
			print(len(PM1_2))
			print(sorted(PM1_2))
			break

	intsct_mask = np.zeros((145))
	for i, genome in enumerate(row_names):
		if genome in PM1_2:
			intsct_mask[i] = 1
	intsct_mask = [i==1 for i in intsct_mask]

	all_feats_matrix = all_feats_matrix[intsct_mask]
	row_names = row_names[intsct_mask]
	print('pm1_2:',all_feats_matrix)
	print('pm1_2:',all_feats_matrix.shape)


	feat_mask = np.zeros(len(all_feats))
	print('before feat clean:',all_feats_matrix.shape)
	all_feats_matrix = np.transpose(all_feats_matrix)
	for i, col in enumerate(all_feats_matrix):
		for ele in col:
			if(ele==-1):
				feat_mask[i] = 1
				break
	feat_mask = [i==0 for i in feat_mask]
	all_feats_matrix = all_feats_matrix[feat_mask]
	all_feats_matrix = np.transpose(all_feats_matrix)
	all_feats = all_feats[feat_mask]
	print('after feat clean:',all_feats_matrix)
	print('after feat clean:',all_feats_matrix.shape)
	print('row names:', len(row_names))
	print('feat names:', len(all_feats))

	np.save('data/unfiltered/omnilog_matrix.npy', all_feats_matrix)
	np.save('data/unfiltered/omnilog_rows.npy', row_names)
	np.save('data/unfiltered/omnilog_cols.npy', all_feats)





	#omni_copy = deepcopy(omnilog_cols)

	"""
	drug_count_array = omni_df.values[:,1]
	for i, row in enumerate(omni_df.values):
		if row[0] not in row_names:
			drug_count_array[i]='NULL'
	counts = collections.Counter(drug_count_array)
	#print(counts)

	#

	max = 0

	for genomeid in row_names:
		count=0
		for row in omni_df.values:
			if(row[0]==genomeid):
				count+=1
		if (count>max):
			print('highest is now', count, genomeid)
			max = count

	#

	num_rows = len(row_names)
	num_cols = len(omnilog_cols)
	#pm12 mask


	omnilog_matrix = np.zeros((num_rows,num_cols), dtype='object')

	for row in omni_df.values:
		row_index = find_index(row[0], row_names)
		col_index = find_index(row[1], omnilog_cols)
		if(row_index ==-1):
			#print("{} was not seen".format(row[1]))
			continue
		omnilog_matrix[row_index][col_index] = row[2]
	print(omnilog_matrix)
	print(omnilog_matrix.shape)
	"""
