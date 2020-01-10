
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
	#turns ECI-93 into ECI-0093
	if (genomeid[:3]=='ECI'):
		eci, num = genomeid.split('-')
		return'{0}-{1:0>4}'.format(eci, num)
	else:
		return genomeid

def find_eci(genomeid):
	if(genomeid[:8] == 'complete'):
		return genomeid[-11:-4]+'-contigs-all'
	try:
		eci = re.search(r'(ECI-\d{1,4})|(ON-2011)|(Sakai)|(EDL933)', genomeid).group()
	except:
		print("no regex match for eci id in genome", genomeid)
		raise
	return eci
	#[\(ECI\-\d{1,4}\)\(ON\-2011\)\(Sakai\)]

def intersection(lst1, lst2):
	temp = set(lst2)
	lst3 = [value for value in lst1 if value in temp]
	return lst3

def get_PM_substrates():
	"""
	Returns a dictionary { PM# : list of substrates}
	"""

	pms = {}

	with open("data/omnilog_well_descriptions.txt") as file:
		cur_pm = ''
		for n, line in enumerate(file):
			line = line.rstrip()
			try:
				if line[0:2] == 'PM':
					cur_pm = line
					pms[cur_pm] = []
				else:
					pms[cur_pm].append(line[5:-1])
			except:
				print("Failure reading line {}: {}".format(n, line))
				raise

	return pms

if __name__ == "__main__":
	"""
	We are going to scan to find largest subset of omnilog substrates that is consistent across the most genomes
	We are then going to build a 2D matrix with rows labelled as genomes and the columns are labeled as substrates
	Next we go through the data and load the [row][col] with the area under the curve value for that substrate genome combo
	We save this so that we can later replace the genome name with some form of training data
	"""
	row_names = pd.read_csv('data/final_omnilog_metadata.csv')
	row_names = row_names.values[:,0]
	omni_df = pd.read_csv('data/omnilog_data_summary.txt', delimiter = '\t')
	#pm = pd.read_csv('data/omnilog_well_descriptions.txt', delimiter = '\t')

	omnilog_cols = np.empty(192,dtype = 'object')
	all_feats = np.unique(omni_df.values[:,1])
	#print(all_feats)
	#print(find_index('Sucrose', all_feats))
	all_feats_matrix = np.zeros((len(row_names),len(all_feats)), dtype='object')
	all_feats_matrix[:] = -1

	coutnt=0
	for i, row in enumerate(omni_df.values):
		row[0] = pad_zeros(row[0])
		if len(row[0])==7 and row[0][2]=='-' and row[0]!= 'ON-2011':
			rowindx = find_index(row[0]+"-contigs-all", row_names)
		else:
			rowindx = find_index(row[0], row_names)
		colindx = find_index(row[1], all_feats)
		#print(rowindx, colindx)
		if(rowindx != -1 and colindx != -1):
			all_feats_matrix[rowindx,colindx] = row[2]
			coutnt +=1
		else:
			raise Exception("Could not find col or row for {} ({},{})".format(row, rowindx, colindx))
	#print(coutnt)
	#print('start:',all_feats_matrix)
	#print('start:',all_feats_matrix.shape)

	"""
		Matrix currently has all possible features against all possible genomes,
		features from omnilog_data_summary, genomes from omnilog_metadata
	"""

	first = []
	PM1_2 = []
	for i in range(1,21):
		#print('***** PM'+str(i)+' *****')
		PM1 = listdir('data/PM'+str(i))
		PM1 = [find_eci(i) for i in PM1]
		PM1 = [pad_zeros(i) for i in PM1]
		if(i==1):
			first = PM1
		#print(sorted(PM1))
		if(i==2):
			#print('***** INTERSECTION *****')
			PM1_2 = intersection(PM1, first)
			#print(len(PM1_2))
			#print(sorted(PM1_2))
			break

	print("Shape prior to intersection mask", all_feats_matrix.shape)
	intsct_mask = np.zeros((all_feats_matrix.shape[0]))
	for i, genome in enumerate(row_names):
		if genome in PM1_2:
			intsct_mask[i] = 1
		else:
			print("genome {} not found in both PM1 and PM2".format(genome))
	intsct_mask = [i==1 for i in intsct_mask]

	all_feats_matrix = all_feats_matrix[intsct_mask]
	row_names = row_names[intsct_mask]
	#print('pm1_2:',all_feats_matrix)
	#print('pm1_2:',all_feats_matrix.shape)


	pms = get_PM_substrates()

	keep = [i in pms['PM1'] or i in pms['PM2'] for i in all_feats]
	print("num_feats:", len(all_feats))
	print("num of feats in pm1 or 2:",np.sum(keep))

	invalid_PM1or2 = []

	feat_mask = np.zeros(len(all_feats))
	print('before feat clean:',all_feats_matrix.shape)
	all_feats_matrix = np.transpose(all_feats_matrix)
	for i, col in enumerate(all_feats_matrix):
		for j, ele in enumerate(col):
			if(ele==-1):
				feat_mask[i] = 1
				if all_feats[i] in pms['PM1'] or all_feats[i] in pms['PM2']:
					invalid_PM1or2.append(all_feats[i])
					#raise Exception('Feature {} is invalid for genome {}'.format(all_feats[i],row_names[j]))
				break
	print("Missing features:", invalid_PM1or2)
	feat_mask = [i==0 for i in feat_mask]
	all_feats_matrix = all_feats_matrix[feat_mask]
	all_feats_matrix = np.transpose(all_feats_matrix)
	all_feats = all_feats[feat_mask]
	#print('after feat clean:',all_feats_matrix)
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
