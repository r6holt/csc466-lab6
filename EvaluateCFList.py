# CSC466 - Lab 6
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
# Adam Havstad (ahavstad@calpoly.edu)
import sys
import numpy as np
import pandas as pd
from importer import Importer
import classifier
import json
import string
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from random import randint
import csv
import parser
import EvaluateCFRandom as ecr


def main():
	if len(sys.argv) == 3:
		method = int(sys.argv[1])
		filename = sys.argv[2]

	else:
		print("syntax: python3 EvaluateCFList.py <method (int)> <filenaem>")
		print("Methods: ")
		print("\t0 : dummy")
		print("\t1 : Mean Utility")
		print("\t2 : Weighted Sum")
		print("\t3 : Weighted nNN")
		print("\t4 : Average nNN")
		return

	#all parsing should be done here
	path = 'data/jester-data-1.csv'
	parsed = parser.Parse(path)

	pairs = pd.read_csv(filename, header=None, names=['uid', 'iid'])

	#prep the output files
	out = open("out_method_{}_list.csv".format(method), "w")
	csv_out = csv.writer(out, delimiter=',', quotechar='"')
	csv_out.writerow(['userId', 'itermID', 'Actual_Rating', 'Predicted_Rating', 'Delta_Rating'])

	evaluate(parsed, method, pairs, csv_out)


def evaluate(parsed, method, pairs_df, out):
	'''
	if n >= 2:
		pass
	elif n == 0 or n == 0: 
		print("No Cross validation")
		return
	elif n == -1:
		print("all but one")
		n = df.shape[0]
	

	n = parsed.ratings.shape[0]
	kf = KFold(n_splits=n, shuffle=True)
	'''

	#labels = parsed.get_labels()
	labels = ['T', 'F']
	df = parsed.get_df()
	#open("results.txt", "w")

	#conf_list = []
	
	'''
	for trainidx, testidx in kf.split(df):
		train = df.iloc[trainidx]
		test =  df.iloc[testidx]
	'''

	pairs = []
	for idx, row in pairs_df.iterrows():
		uid = row['uid']
		iid = row['iid']
		#print(rand_uid, rand_iid)
		if df.iloc[uid, iid] != 99:
			pairs += [(uid, iid)]
			


	#change these to the actual name of each method
	if(method == 0):
		conf, errors = ecr.get_confusion_method_0(parsed, pairs, out)



	#if method ...	

	else:
		print("no matching method")
		return
	
	ecr.print_output([conf], labels, errors)


if __name__ == '__main__':
	main()