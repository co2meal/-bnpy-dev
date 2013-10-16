import sys
sys.path.append('../../bnpy/')

from random import random
import bnpy
import Learn
import numpy as np
import pylab as pl

def main():
    X = np.loadtxt('../demodata/faithful.txt')
    data = bnpy.data.XData(X)
    N = 50 # number of runs of each algorithms for a given K
    maxK = 7 # K ranges from 1 to maxK

    # emscore stores log-likelihood of data from EM runs
    # emaverage stores the average of emscores different K's
    emscore = np.zeros((N, maxK-1))
    emaverage = np.zeros(maxK-1)

    # vbscore stores variational lower bound from VB runs
    # vbaverage stores average of vbscores different K's
    vbscore = np.zeros((N, maxK-1))
    vbaverage = np.zeros(maxK-1)

    # EM
    for k in range(1, maxK):
        print 'fitting with EM @ K = ' + str(k)
        for i in range(N):
            hmodel, LP, evBound = Learn.run(data, 'MixModel', 'Gauss', 'EM', K=k, jobname=str(i) + 'GMMCompareK')
            emscore[i, k-1] = evBound
        emaverage[k-1] = np.average(emscore[:,k-1])

    print 'EM average log-likelihood:'
    print emaverage

    # VB
    for k in range(1, maxK):
        print 'fitting with EM @ K = ' + str(k)
        for i in range(N):
            hmodel, LP, evBound = Learn.run(data, 'MixModel', 'Gauss', 'VB', K=k, jobname=str(i) + 'GMMCompareK')
            vbscore[i, k-1] = evBound
        vbaverage[k-1] = np.average(vbscore[:,k-1])

    print 'VB average lower bound:'
    print vbaverage

    x1 = range(1, maxK)
    x2 = range(1, maxK)

    # plot performance criteria vs. K
    pl.figure(0)
    pl.title('EM plot')
    for K in range(1, maxK):
        pl.scatter(np.ones((N, 1)) * K, emscore[:, K-1], marker='+')

    pl.figure(1)
    pl.title('VB plot')
    for K in range(1, maxK):
        pl.scatter(np.ones((N, 1)) * K, vbscore[:, K-1], marker='+')
    pl.show()
    
    return


if __name__ == '__main__':
    main()
