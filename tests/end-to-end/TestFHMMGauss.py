'''
TestFHMMGauss.py

Tests for FiniteHMM.py using GaussObsModel.py for the observation model
'''

import numpy as np
import unittest

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest
import Util

#class TestSimple(AbstractEndToEndTest):
#    print 'lol'

class TestHMMK4_EM(AbstractEndToEndTest):
    ''' Simple example with K=4 components 
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
        sigmas[0,:,:] = np.asarray([[1, 0], [0, 1]])
        sigmas[1,:,:] = np.asarray([[1, 0], [0, 1]])
        sigmas[2,:,:] = np.asarray([[1, 0], [0, 1]])
        sigmas[3,:,:] = np.asarray([[1, 0], [0, 1]])

        #transPi = np.asarray([[0.0, 0.5, 0.0, 0.5], \
        #                          [0.25, 0.0, 0.5, 0.25], \
        #                          [0.0, 0.0, 0.0, 1.0], \
        #                          [1.0, 0.0, 0.0, 0.0]])

        transPi = np.asarray([[0.0, 1.0, 0.0, 0.0], \
                                  [0.0, 0.0, 1.0, 0.0], \
                                  [0.0, 0.0, 0.0, 1.0], \
                                  [1.0, 0.0, 0.0, 0.0]])



        initState = 1
        initPi = np.zeros(self.K)
        initPi[initState] = 1

        self.TrueParams = dict(K = self.K, m = mus, L = sigmas,
                                   transPi = transPi)
        self.ProxFunc = dict(L = Util.CovMatProxFunc,
                             m = Util.VectorProxFunc,
                             transPi = Util.ProbMatrixProxFunc,
                             initPi = Util.ProbVectorProxFunc)

        #Generate gaussian data, transitioning between states using transPi and
          #starting in initState
        nObsTotal = 25000
        prng = np.random.RandomState()
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

        #Basic configuration
        self.allocModelName = 'FiniteHMM'
        self.obsModelName = 'Gauss'
        self.kwargs = dict(nLap = 30, K = self.K, initAlpa = .01)
        self.fromScratchTrials = 5
        self.fromScratchSuccessRate = .8
        #self.atol = 8
        self.learnAlgs = ['EM']

        # Substitute config used for "from-scratch" tests only
        #  anything in here overrides defaults in self.kwargs
        self.fromScratchArgs = dict(nLap=40, K=self.K, initname='randexamples')
    
