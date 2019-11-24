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

def knnHelper(corr_matrix, k, source_user_ids):
    user_knn = {}
    for uid in source_user_ids:
        user = corr_matrix[uid]
        sorted_neighbors = sorted(range(len(user)), key=lambda k: 1-abs(user[k]))
        user_knn[uid] = sorted_neighbors[1:]
    return user_knn

def meanUtility(matrix, jokes, users, uid, iid):
    user = users[uid]
    joke = jokes[iid]
    prev_rating = matrix[uid, iid]
    if prev_rating != 99:
        matrix[uid, iid] = 99
    pred_rating = joke.mean_rating
    if pred_rating > 10:
        pred_rating = 10
    elif pred_rating < -10:
        pred_rating = -10
    matrix[uid, iid] = prev_rating
    return pred_rating

def weightedSum(matrix, jokes, users, corr_matrix, uid, iid):
    user = users[uid]
    joke = jokes[iid]
    prev_rating = matrix[uid, iid]
    if prev_rating != 99:
        matrix[uid, iid] = 99
    sum_of_weights = 0
    sum_of_products = 0
    for i, user_rating in enumerate(joke.ratings):
        if user_rating != 99:
            sim = corr_matrix[uid, i]
            sum_of_weights += abs(sim)
            sum_of_products += sim * user_rating
    pred_rating = (1/sum_of_weights) * sum_of_products
    if pred_rating > 10:
        pred_rating = 10
    elif pred_rating < -10:
        pred_rating = -10
    matrix[uid, iid] = prev_rating
    return pred_rating


def knnWeighted(matrix, jokes, users, corr_matrix, uid_knn, k, uid, iid):
    user = users[uid]
    joke = jokes[iid]
    prev_rating = matrix[uid, iid]
    if prev_rating != 99:
        matrix[uid, iid] = 99
    sum_of_weights = 0
    sum_of_products = 0
    knn = uid_knn[uid]
    num_neighbors_added = 0
    for neighborId in knn:
        if num_neighbors_added == k:
            break
        user_rating = joke.ratings[neighborId]
        if user_rating != 99:
            sim = corr_matrix[uid, neighborId]
            sum_of_weights += abs(sim)
            sum_of_products += sim * user_rating
            num_neighbors_added += 1
    pred_rating = (1/sum_of_weights) * sum_of_products
    if pred_rating > 10:
        pred_rating = 10
    elif pred_rating < -10:
        pred_rating = -10
    matrix[uid, iid] = prev_rating
    return pred_rating


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
