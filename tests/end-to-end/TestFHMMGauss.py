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
        sigmas[0,:,:] = np.asarray([[2, 0], [0, 2]])
        sigmas[1,:,:] = np.asarray([[2, 0], [0, 2]])
        sigmas[2,:,:] = np.asarray([[2, 0], [0, 2]])
        sigmas[3,:,:] = np.asarray([[2, 0], [0, 2]])

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

        #Generate gaussian data, transitioning between states using transPi and
          #starting in initState
        seqLens = ((6000,6000,6000,6000,1000))
        seqInds = list([0])
        seed = np.random.randint(0, sys.maxint) 
        prng = np.random.RandomState(seed)

        Z = ()
        X = None
            
        for j in xrange(len(seqLens)):

            initState = prng.multinomial(1, initPi)
            initState = np.nonzero(initState)[0][0]

            seqZ = list()
            seqX = list()
            seqZ.append(initState)
            seqX.append(np.random.multivariate_normal(mus[seqZ[0],:], \
                                                          sigmas[seqZ[0],:,:]))
        
            for i in xrange(seqLens[j] - 1):
                trans = prng.multinomial(1, transPi[seqZ[i]])
                nextState = np.nonzero(trans)[0][0]
                seqZ.append(nextState)
                seqX.append(np.random.multivariate_normal(mus[nextState,:], 
                                                          sigmas[nextState,:,:]))
            Z = np.append(Z, seqZ)
            if X is None:
                X = seqX
            else:
                X = np.vstack((X, seqX))
            seqInds.append(seqLens[j] + seqInds[j])
            
        self.Data = bnpy.data.SeqXData(X, seqInds, Z)
        
        self.TrueParams = dict(K = self.K, m = mus, Sigma = sigmas)
                                   #transPi = transPi)
        self.ProxFunc = dict(Sigma = Util.CovMatProxFunc,
                             m = Util.VectorProxFunc,
                             #transPi = Util.ProbMatrixProxFunc,
                             initPi = Util.ProbVectorProxFunc,
                             canttestthis = Util.ProbVectorProxFunc)

        #Basic configuration
        self.allocModelName = 'FiniteHMM'
        self.obsModelName = 'Gauss'
        self.kwargs = dict(nLap = 20, K = self.K, initAlpa = .01)
        self.fromScratchTrials = 5
        self.fromScratchSuccessRate = .4
        self.learnAlgs = ['EM']

        # Substitute config used for "from-scratch" tests only
        #  anything in here overrides defaults in self.kwargs
        #Note nLap=15 is all that is necessary -- a successful run almost always
        # converges in under 10 iterations
        self.fromScratchArgs = dict(nLap=15, K=self.K, initname='randexamples', 
                                    min_covar=1e-8, init_min_covar = .01, 
                                    initAlpha = .01)


            
            
    
class TestHMMK4_VB(TestHMMK4_EM):
    

    __test__ = True

    def setup(self):
        super(TestHMMK4_VB, self).setup()
        self.learnAlgs = ['VB', 'soVB', 'moVB']

        
        
