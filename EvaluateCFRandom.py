# CSC466 - Lab 3
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
import sys
import numpy as np
from importer import Importer
import classifier
import json
import string
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from random import randint
import csv
import parser



def main():
	if len(sys.argv) == 4:
		method = int(sys.argv[1])
		size = int(sys.argv[2])
		repeats = int(sys.argv[3])

	else:
		print("syntax: python3 EvaluateCFRandom.py <method (int)> <size (int)> <repeats (int)>")
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

	#prep the output files
	out = open("out_method_1.csv", "w")
	csv_out = csv.writer(out, delimiter=',', quotechar='"')
	csv_out.writerow(['userId', 'itermID', 'Actual_Rating', 'Predicted_Rating', 'Delta_Rating'])

	for _ in range(repeats):
		evaluate(parsed, method, size, csv_out)


def evaluate(parsed, method, size, out):
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
	count = 0
	'''
	for trainidx, testidx in kf.split(df):
		train = df.iloc[trainidx]
		test =  df.iloc[testidx]
	'''
	pairs = []
	while count < size:
		while 1:
			rand_uid = randint(0, parsed.users-1)
			rand_iid = randint(0, parsed.jokes-1)
			#print(rand_uid, rand_iid)
			if df.iloc[rand_uid, rand_iid] != 99:
				pairs += [(rand_uid, rand_iid)]
				count+=1
				break


	if(method == 0):
		conf, errors = get_confusion_method_0(parsed, pairs, out)
#		conf_list += [conf]		

	else:
		print("no matching method")
		return
	
	print_output([conf], labels, errors)#, n)

	#print_output(conf_list, labels, n)

		#print(conf)


def get_confusion_method_0(parsed, pairs, out_csv):
	pred = []
	true = []
	errors = []
	# output = open("output/{}-c45.txt".format(out), "w")

	for uid, iid in pairs:
		val = classifier.Classify.classify_method_0(parsed, uid, iid)
		real = parsed.ratings.iloc[uid, iid]
		error = abs(real - val)
		
		if val >= 5:
			prec = "Recommended"
		else:
			prec = "Not Recommended"

		if real >= 5:
			trec = "Recommended"
		else: 
			trec = "Not Recommended"

		pred += [prec]
		true += [trec]
		errors += [error]
		out_csv.writerow([uid, iid, real, val, error])

	return confusion_matrix(true, pred, ['Recommended', 'Not Recommended']), errors


def add_confusions(conf_list):
	#print(conf_list)
	if len(conf_list) == 0:
		return None

	overall_conf = conf_list[0]
	for i in range(1, len(conf_list)):
		#print("adding conf: \n\t", overall_conf, "\n\t+\n\t",conf_list[i])
		overall_conf = np.add(overall_conf, conf_list[i])
		

	return overall_conf

def print_confusion(conf, labels):
	#print(type(conf))
	#labels = ["T", "F"]
	print("  ", end=" ")
	for label in labels:
		print(label, end = " ")

	print()
	for i in range(len(labels)):
		print(labels[i], conf[i])

def precision_from_conf(conf, labels):
	correct = 0
	total = 0
	binary = False

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			#print("conf: ", conf)
			count = conf[i][j]
			if(not binary or (binary and j==0)):
				total += conf[i][j]

			if (not binary and (i==j)) or (binary and i==0 and j==0):
				correct += count

	if total == 0:
		return 0

	return correct/total

def recall_from_conf(conf, labels):
	correct = 0
	total = 0
	binary = False

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			count = conf[i][j]
			if(not binary or (binary and i==0)):
				total += conf[i][j]

			if (not binary and (i==j)) or (binary and i==0 and j==0):
				correct += count

	if total == 0:
		return 0

	return correct/total

def pf_from_conf(conf, labels):
	incorrect = 0
	total = 0
	binary = False

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			count = conf[i][j]
			if(not binary or (binary and i==1)):
				total += conf[i][j]

			if (not binary and (i==j)) or (binary and i==1 and j==0):
				incorrect += count
	if total == 0:
		return 0

	return incorrect/total

def fmes_from_stats(precision, recall):
	numer = 2 * precision * recall
	denom = precision + recall
	if(denom == 0):
		return 0
	fmes = numer / denom
	return fmes

def accuracy_from_conf(conf, labels):
	correct = 0
	total = 0

	n = len(labels)
	if n == 2:
		binary = True

	for i in range(n):
		for j in range(n):
			count = conf[i][j]
			total += conf[i][j]

			if (i==j):
				correct += count

	if total == 0:
		return 0

	return correct/total

def mae_from_errors(errors):
	n = len(errors)
	total = 0
	for error in errors:
		total += error

	return total/n

def print_output(conf_list, labels, errors):#, n):
	print("--------------------------")
	print("\tVALIDATION")
	print("--------------------------")
	#print("Folds:  {}\n".format(n))
	overall_conf = add_confusions(conf_list)
	#print("overall_conf", overall_conf)
	print("Confusion Matrix: ")
	print_confusion(overall_conf, labels)
	precision = precision_from_conf(overall_conf, labels)
	print("Precision: ", precision)
	recall = recall_from_conf(overall_conf, labels)
	print("Recall: ", recall)
	#pf = pf_from_conf(overall_conf, labels)
	#print("pf: ", pf)
	fmes = fmes_from_stats(precision, recall)
	print("F-measure: ", fmes)
	
	overall_accuracy = accuracy_from_conf(overall_conf, labels)
	print("Overall Accuracy: ", overall_accuracy)

	mae = mae_from_errors(errors)
	print("\nMAE: ", mae)
	print("\n\n")
	#acc_list = []

	#for conf in conf_list:
	#	acc_list += [accuracy_from_conf(conf, labels)]

	#average_accuracy = np.mean(acc_list)
	#print("Average Accuracy: ",  average_accuracy)
	#print(acc_list)
	#print("Overall Error Rate: ", 1-overall_accuracy)
	#print("Average Error Rate: ", 1-average_accuracy)

if __name__ == '__main__':
	main()







