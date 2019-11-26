import col_filtering
import sys, argparse, statistics, random

def getTests(matrix, fpath):
    tests = []
    with open(fpath) as f:
        for l in f:
            l = l.split(', ')
            uid = int(l[0])
            iid = int(l[1])
            if matrix[uid, iid] != 99:
                tests.append((uid, iid))
    return tests

def evaluate(meth, test, source_uids, matrix, jokes, users, corr_matrix):
    if meth == 'm':
        pred = col_filtering.meanUtility(matrix, jokes, users, test[0], test[1])
    elif meth == 'w':
        pred = col_filtering.weightedSum(matrix, jokes, users, corr_matrix, test[0], test[1])
    elif meth == 'knnw':
        k = 1000 
        uid_knn = col_filtering.knnHelper(corr_matrix, k, source_uids)
        pred = col_filtering.knnWeighted(matrix, jokes, users, corr_matrix, uidknn, k, test[0], test[1])
    else:
        k = 1000
        uid_knn = col_filtering.knnHelper(corr_matrix, k, source_uids)
        pred = col_filtering.knnAverage(matrix, jokes, users, corr_matrix, uidknn, k, test[0], test[1])
    return pred

def getArgs():
    p = argparse.ArgumentParser(description='EvaluateCFList')
    p.add_argument('-f', '--fpath', help='joke data file path', required=True)
    p.add_argument('-m', '--meth', help='method: m | w | knnw | knna', required=True)
    p.add_argument('-t', '--fpath2', help='test file path', required=True)
    return vars(p.parse_args())

def main():
    args = getArgs()
    fpath = args['fpath']
    meth = args['meth']
    fpath2 = args['fpath2']
    if meth != 'm' and meth != 'w' and meth != 'knnw' and meth != 'knna':
        print("Your method is flawed.")
        sys.exit(0)
    matrix, jokes, users = col_filtering.parse(fpath)
    corr_matrix = col_filtering.pearsonCorr(users)
    tests = getTests(matrix, fpath2)
    source_uids = [test[0] for test in tests]
    tae = 0
    for test in tests:
        pred = evaluate(meth, test, source_uids, matrix, jokes, users, corr_matrix)
        obs = matrix[test[0], test[1]]
        ae = abs(pred-obs)
        print(f"{test[0]}, {test[1]}, {obs}, {pred}, {ae}")
        tae += ae
    mae = tae / len(tests)
    print(mae)

if __name__ == '__main__':
    main()
