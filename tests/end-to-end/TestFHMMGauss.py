'''
TestFHMMGauss.py

Tests for FiniteHMM.py using GaussObsModel.py for the observation model
'''

import numpy as np
import unittest
import sys

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest
import Util


class TestHMMK4_EM(AbstractEndToEndTest):
    ''' Simple example with K=4 components, all with very separated means
    '''
    __test__ = True

    def setUp(self):



        #Create true parameters
        self.K = 4
        mus = np.asarray([[0, 0], \
                  [0, 10], \
                  [10, 0], \
                  [10, 10]])

        sigmas = np.empty((4,2,2))
        sigmas[0,:,:] = np.asarray([[2, 0], [0, 2]])
        sigmas[1,:,:] = np.asarray([[2, 0], [0, 2]])
        sigmas[2,:,:] = np.asarray([[2, 0], [0, 2]])
        sigmas[3,:,:] = np.asarray([[2, 0], [0, 2]])


        transPi = np.asarray([[0.0, 1.0, 0.0, 0.0], \
                                  [0.0, 0.0, 1.0, 0.0], \
                                  [0.0, 0.0, 0.0, 1.0], \
                                  [1.0, 0.0, 0.0, 0.0]])



        initState = 1
        initPi = np.zeros(self.K)
        initPi[initState] = 1

        #Generate gaussian data, transitioning between states using transPi and
          #starting in initState
        nObsTotal = 25000
        seed = np.random.randint(0, sys.maxint) 
        prng = np.random.RandomState(seed)
        Z = list()
        X = list()
        Z.append(initState)
        X.append(np.random.multivariate_normal(mus[Z[0],:], sigmas[Z[0],:,:]))

        for i in xrange(nObsTotal-1):
            trans = prng.multinomial(1, transPi[Z[i]])
            nextState = np.nonzero(trans)[0][0]
            Z.append(nextState)
            X.append(np.random.multivariate_normal(mus[nextState,:], 
                                                   sigmas[nextState,:,:]))
            
        Z = np.asarray(Z)
        X = np.vstack(X)
        self.Data = bnpy.data.XData(X)

       #FiniteHMM finds precision matricies, so convert ours to precision
        for k in xrange(self.K):
            sigmas[k,:,:] = np.linalg.inv(sigmas[k,:,:])

        self.TrueParams = dict(K = self.K, m = mus, L = sigmas,
                                   transPi = transPi)
        self.ProxFunc = dict(L = Util.CovMatProxFunc,
                             m = Util.VectorProxFunc,
                             transPi = Util.ProbMatrixProxFunc,
                             initPi = Util.ProbVectorProxFunc)

        #Basic configuration
        self.allocModelName = 'FiniteHMM'
        self.obsModelName = 'Gauss'
        self.kwargs = dict(nLap = 20, K = self.K, initAlpa = .01)
        self.fromScratchTrials = 5
        self.fromScratchSuccessRate = .4
        self.learnAlgs = ['EM']

        # Substitute config used for "from-scratch" tests only
        #  anything in here overrides defaults in self.kwargs
        #Note nLap=15 is all that is necessary, as a successful run almost always
        # converges in under 10 iterations
        self.fromScratchArgs = dict(nLap=15, K=self.K, initname='randexamples', 
                                    min_covar=1e-8, initAlpha = .01)
    
