import argparse
import pandas as pd
import numpy as np
import math
import queue

import time

class User:
    def __init__(self, num_ratings, ratings):
        self.num_ratings = num_ratings
        self.ratings = ratings
        self.num_ratings = 0
        self.sum_ratings = 0
        for r in ratings:
            if r != 99:
                self.num_ratings += 1
                self.sum_ratings += r
        self.mean_rating = self.sum_ratings / self.num_ratings

class Joke:
    def __init__(self, ratings):
        self.ratings = ratings
        self.num_users = len(self.ratings)
        self.num_ratings = 0
        self.sum_ratings = 0
        for r in ratings:
            if r != 99:
                self.num_ratings += 1
                self.sum_ratings += r
        self.mean_rating = self.sum_ratings / self.num_ratings
        self.inverse_user_freq = math.log2(self.num_users / self.num_ratings)

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

def knnHelper(corr_matrix, k, source_uids):
    user_knn = {}
    for uid in source_uids:
        u = corr_matrix[uid]
        neighbors = sorted(range(len(u)), key=lambda k: 1-abs(u[k]))
        user_knn[uid] = neighbors[1:]
    return user_knn

def meanUtility(matrix, jokes, users, uid, iid):
    u = users[uid]
    j = jokes[iid]
    prev = matrix[uid, iid]
    if prev != 99:
        matrix[uid, iid] = 99
    pred = j.mean_rating
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
    kwrating = knnWeighted(matrix, jokes, users, corr_matrix, uid_knn, k, 2, 5)
    print(kwrating)

if __name__ == '__main__':
    main()
