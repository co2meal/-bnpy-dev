import sys
sys.path.append('../bnpy/')

from random import random
import bnpy
import Learn
import numpy as np
import pylab as pl

def main():
    X = np.loadtxt('../EM-VB-comparison/faithful.txt')
    data = bnpy.data.XData(X)
    N = 50
    maxK = 7

    gscore = np.zeros((N, maxK-1))
    gaverage = np.zeros(maxK-1)
    vbscore = np.zeros((N, maxK-1))
    vbaverage = np.zeros(maxK-1)

    # EM
    for k in range(1, maxK):
        print 'fitting with EM @ K = ' + str(k)
        for i in range(N):
            hmodel, LP, evBound = Learn.run(data, 'MixModel', 'Gauss', 'EM', K=k, jobname=str(random()))
            gscore[i, k-1] = evBound
        gaverage[k-1] = np.average(gscore[:,k-1])

    print 'EM average log-likelihood:'
    print gaverage

    # VB
    for k in range(1, maxK):
        print 'fitting with EM @ K = ' + str(k)
        for i in range(N):
            hmodel, LP, evBound = Learn.run(data, 'MixModel', 'Gauss', 'VB', K=k, jobname=str(random()))
            vbscore[i, k-1] = evBound
        vbaverage[k-1] = np.average(vbscore[:,k-1])

    print 'VB average log-likelihood:'
    print vbaverage

    x1 = range(1, maxK)
    x2 = range(1, maxK)

    pl.figure(0)
    pl.title('EM plot')
    for K in range(1, maxK):
        pl.scatter(np.ones((N, 1)) * K, gscore[:, K-1], marker='+')

    pl.figure(1)
    pl.title('VB plot')
    for K in range(1, maxK):
        pl.scatter(np.ones((N, 1)) * K, vbscore[:, K-1], marker='+')
    pl.show()
    
    return



if __name__ == '__main__':
    main()
