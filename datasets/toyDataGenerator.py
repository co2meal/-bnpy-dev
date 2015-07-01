__author__ = 'roie'

import bnpy
import numpy as np
import matplotlib
from matplotlib import pylab
import math

def genData():
    # 7524
    PRNG = np.random.RandomState(7524)

    counter = 0
    nClusters = 50
    nDataPts = 10000
    dpCount = 0
    fracCount = 0.0
    trueLabels = None
    data = None

    while (counter < nClusters):
        if (counter == nClusters - 1):
            n = nDataPts - dpCount
        else:
            frac = PRNG.rand()*min((1-fracCount),0.5)
            fracCount = frac+fracCount
            n = frac*nDataPts
            dpCount = dpCount + n

        center = [PRNG.rand(), PRNG.rand()]

        cov = np.matrix(PRNG.rand(2,2))
        cov = np.transpose(cov)*cov/100
        m = int(math.ceil(n))
        clusterData = PRNG.multivariate_normal(center, cov, m)

        # pylab.plot(clusterData[:,0], clusterData[:,1], '.', color=[PRNG.rand(),PRNG.rand(),PRNG.rand()]);
        if (counter == 0):
            data = clusterData
            trueLabels = counter*np.ones(m)
        else:
            data = np.vstack([data, clusterData])
            trueLabels = np.hstack([trueLabels, counter*np.ones(m)])

        counter = counter + 1

    # pylab.gcf().set_size_inches(4, 4);
    # pylab.axis('image'); pylab.xlim([-0.5, 1.5]); pylab.ylim([-0.5, 1.5]);
    # pylab.show()

    DataObj = bnpy.data.XData(data)
    DataObj.TrueParams = dict()
    DataObj.TrueParams['Z'] = trueLabels
    return DataObj
