# CSC466 - Lab 6
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
# Adam Havstad (ahavstad@calpoly.edu)

import argparse, math, queue, time
import pandas as pd
import numpy as np

class User:
	def __init__(self, rcount, ratings):
		self.rcount = rcount
		self.ratings = ratings
		self.rcount = 0
		self.rsum = 0
		for r in ratings:
			if r != 99:
				self.rcount += 1
				self.rsum += r
		self.ravg = self.rsum / self.rcount

class Joke:
	def __init__(self, ratings):
		self.ratings = ratings
		self.ucount = len(self.ratings)
		self.rcount = 0
		self.rsum = 0
		for r in ratings:
			if r != 99:
				self.rcount += 1
				self.rsum += r
		self.ravg = self.rsum / self.rcount
		self.ifuser = math.log2(self.ucount / self.rcount)

def meanUtility(matrix, jokes, users, uid, iid):
	u = users[uid]
	j = jokes[iid]
	prev = matrix[uid, iid]
	if prev != 99:
		matrix[uid, iid] = 99
	pred = j.ravg
	if pred > 10:
		pred = 10
	elif pred < -10:
		pred = -10
	matrix[uid, iid] = prev
	return pred

def weightedSum(matrix, jokes, users, corr_matrix, uid, iid):
	u = users[uid]
	j = jokes[iid]
	prev = matrix[uid, iid]
	if prev != 99:
		matrix[uid, iid] = 99
	wcount = 0
	pcount = 0
	for i, urating in enumerate(j.ratings):
		if urating != 99:
			sim = corr_matrix[uid, i]
			wcount += abs(sim)
			pcount += sim * urating
	pred = (1/wcount) * pcount
	if pred > 10:
		pred = 10
	elif pred < -10:
		pred = -10
	matrix[uid, iid] = prev
	return pred

def knnHelper(corr_matrix, k, source_uids):
	user_knn = {}
	for uid in source_uids:
		u = corr_matrix[uid]
		neighbors = sorted(range(len(u)), key=lambda k: 1-abs(u[k]))
		user_knn[uid] = neighbors[1:]
	return user_knn

def knnWeighted(matrix, jokes, users, corr_matrix, uid_knn, k, uid, iid):
	u = users[uid]
	j = jokes[iid]
	prev = matrix[uid, iid]
	if prev != 99:
		matrix[uid, iid] = 99
	wcount = 0
	pcount = 0
	knn = uid_knn[uid]
	ncount = 0
	for nid in knn:
		if ncount == k:
			break
		urating = j.ratings[nid]
		if urating != 99:
			sim = corr_matrix[uid, nid]
			wcount += abs(sim)
			pcount += sim * urating
			ncount += 1
	pred = (1/wcount) * pcount
	if pred > 10:
		pred = 10
	elif pred < -10:
		pred = -10
	matrix[uid, iid] = prev
	return pred

def knnAverage(matrix, jokes, users, corr_matrix, uid_knn, k, uid, iid):
	u = users[uid]
	j = jokes[iid]
	prev = matrix[uid, iid]
	if prev != 99:
		matrix[uid, iid] = 99
	pcount = 0
	#print(uid_knn.shape, "\n")
	knn = uid_knn[uid]
	ncount = 0
	for nid in knn:
		if ncount == k:
			break
		urating = j.ratings[nid]
		if urating != 99:
			pcount += urating
			ncount += 1
	pred = (1/k) * pcount
	if pred > 10:
		pred = 10
	elif pred < -10:
		pred = -10
	matrix[uid, iid] = prev
	return pred

def getArgs():
	p = argparse.ArgumentParser(description='Collab Filtering')
	p.add_argument('-f', '--filepath', help='data file path', required=True)
	return vars(p.parse_args())

def parse(filepath):
	df = pd.read_csv(filepath, sep=',', header=None)
	ratings_per_user = df[0]
	df = df.drop(df.columns[0], axis=1)
	jokes = []
	users = []
	for j in df:
		jratings = np.array(df[j])
		jokes.append(Joke(jratings))
	for i, u in df.iterrows():
		uratings = np.array(u)
		users.append(User(ratings_per_user[i], uratings))
	return df.values, jokes, users

def pearsonCorr(users):
	uratings = []
	for u in users:
		ratings = np.array([r if r != 99 else 0 for r in u.ratings])
		uratings.append(ratings)
	corr_matrix = np.corrcoef(uratings)
	return corr_matrix

def main():
	args = getArgs()
	filepath = args['filepath']
	k = 4
	matrix, jokes, users = parse(filepath)
	corr_matrix = pearsonCorr(users)
	uid_knn = knnHelper(corr_matrix, k, [0, 1, 3, 4])
	mrating = meanUtility(matrix, jokes, users, 2, 5)
	print(mrating)
	wrating = weightedSum(matrix, jokes, users, corr_matrix, 2, 5)
	print(wrating)
	kwrating = knnWeighted(matrix, jokes, users, corr_matrix, uid_knn, k, 3, 5)
	print(kwrating)
	karating = knnAverage(matrix, jokes, users, corr_matrix, uid_knn, k, 3, 5)
	print(karating)

if __name__ == '__main__':
	main()
