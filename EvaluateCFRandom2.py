# CSC466 - Lab 6
# Ryan Holt (ryholt@calpoly.edu)
# Grayson Clendenon (gclenden@calpoly.edu)
# Adam Havstad (ahavstad@calpoly.edu)

import col_filtering
import sys, argparse, statistics, random

def getTests(matrix, size):
    tests = []
    count = 0
    ucount = len(matrix)
    icount = len(matrix[0])
    while count < size:
        uid = random.randint(0, ucount-1)
        iid = random.randint(0, icount-1)
        if matrix[uid, iid] != 99:
            tests.append((uid, iid))
            count += 1
    return tests

def evaluate(meth, test, source_uids, matrix, jokes, users, corr_matrix):
    if meth == 'm':
        pred = col_filtering.meanUtility(matrix, jokes, users, test[0], test[1])
    elif meth == 'w':
        pred = col_filtering.weightedSum(matrix, jokes, users, corr_matrix, test[0], test[1])
    elif meth == 'knnw':
        k = 1000 
        uid_knn = col_filtering.knnHelper(corr_matrix, k, source_uids)
        pred = col_filtering.knnWeighted(matrix, jokes, users, corr_matrix, uid_knn, k, test[0], test[1])
    else:
        k = 1000
        uid_knn = col_filtering.knnHelper(corr_matrix, k, source_uids)
        pred = col_filtering.knnAverage(matrix, jokes, users, corr_matrix, uid_knn, k, test[0], test[1])
    return pred

def getArgs():
    p = argparse.ArgumentParser(description='EvaluateCFRandom')
    p.add_argument('-f', '--fpath', help='joke data file path', required=True)
    p.add_argument('-m', '--meth', help='meth: m | w | knnw | knna', required=True)
    p.add_argument('-s', '--size', help='num predictions to eval', required=True)
    p.add_argument('-r', '--reps', help='num test reps', required=True)
    return vars(p.parse_args())

def main():
    args = getArgs()
    fpath = args['fpath']
    meth = args['meth']
    size = int(args['size'])
    reps = int(args['reps'])
    if meth != 'm' and meth != 'w' and meth != 'knnw' and meth != 'knna':
        print("Your meth is flawed.")
        sys.exit(0)
    matrix, jokes, users = col_filtering.parse(fpath)
    corr_matrix = col_filtering.pearsonCorr(users)
    maes = []
    for rep in range(reps):
        print("Rep: {}".format(rep+1))
        tests = getTests(matrix, size)
        source_uids = [test[0] for test in tests]
        tae = 0
        for test in tests:
            pred = evaluate(meth, test, source_uids, matrix, jokes, users, corr_matrix)
            obs = matrix[test[0], test[1]]
            ae = abs(pred-obs)
            print(f"{test[0]}, {test[1]}, {obs}, {pred}, {ae}")
            tae += ae
        mae = tae / size
        maes.append(mae)
        print(mae)
        print("------------------------------------------------------------------")
    print("Mean MAE: {}".format(round(statistics.mean(mae), 3)))
    if len(maes) > 1:
        print("SD MAE: {}".format(statistics.stdev(maes)))
    else:
        print("More reps!")

if __name__ == '__main__':
    main()
