'''
'''
import numpy as np
import copy
from ..distr import DirichletDistr
from ..util import np2flatstr, EPS

from ObsModel import ObsModel

class MultObsModel(ObsModel):
    
  ######################################################### Constructors
  #########################################################
    def __init__(self, inferType, obsPrior=None):
        self.inferType = inferType
        self.obsPrior = obsPrior
        self.comp = list()

    @classmethod
    def CreateWithPrior(cls, inferType, priorArgDict, Data):
        if inferType == 'EM':
            raise NotImplementedError('TODO')
        else:
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        return cls(inferType, obsPrior)

    @classmethod
    def CreateWithAllComps(cls, oDict, obsPrior, compDictList):
        ''' Create MultObsModel, all K component Distr objects, 
            and the prior Distr object in one call
        '''
        if oDict['inferType'] == 'EM':
            raise NotImplementedError("TODO")
        self = cls(oDict['inferType'], obsPrior=obsPrior)
        self.K = len(compDictList)
        self.comp = [None for k in range(self.K)]
        for k in xrange(self.K):
            self.comp[k] = DirichletDistr(**compDictList[k]) 
        return self

  ######################################################### Accessors  
  #########################################################  
    def getElogphiMatrix(self):
      Elogphi = np.empty((self.K, self.comp[0].D))
      for k in xrange(self.K):
        Elogphi[k,:] = self.comp[k].Elogphi
      return Elogphi


  ######################################################### Local Params
  #########################################################   E-step
    def calc_local_params(self, Data, LP, **kwargs):
        ''' Calculate local parameters (E-step)

            Returns
            -------
            LP : bnpy local parameter dict, with updated fields
                E_logsoftev_WordsData : nDistinctWords x K matrix, where
                                entry n,k = log p(word n | topic k)
        '''
        if self.inferType == 'EM':
            raise NotImplementedError('TODO')
        else:
            LP['E_logsoftev_WordsData'] = self.E_logsoftev_WordsData(Data)
        return LP
    
    def E_logsoftev_WordsData(self, Data):
        ''' Return log soft evidence probabilities for each word token.

            Returns
            -------
            E_logsoftev_words : nDistinctWords x K matrix
                                entry n,k gives E log p( word n | topic k)
        '''

        # Obtain matrix where col k = E[ log phi[k] ], for easier indexing
        Elogphi = self.getElogphiMatrix().T.copy()
        E_logsoftev_words = Elogphi[Data.word_id, :]
        return E_logsoftev_words
  

  ######################################################### Suff Stats
  #########################################################
    def get_global_suff_stats(self, Data, SS, LP, **kwargs):
        ''' Calculate and return sufficient statistics.

            Returns
            -------
            SS : bnpy SuffStatDict object, with updated fields
                WordCounts : K x VocabSize matrix
                  WordCounts[k,v] = # times vocab word v seen with topic k
        '''
        wv = LP['word_variational']
        WMat = Data.to_sparse_matrix()
        TopicWordCounts = (WMat * wv).T

        SS.setField('WordCounts', TopicWordCounts, dims=('K','D'))
        SS.setField('N', np.sum(TopicWordCounts,axis=1), dims=('K'))
        return SS

  ######################################################### Global Params
  #########################################################   M-step

    def update_obs_params_EM(self, SS, **kwargs):
        raise NotImplementedError("TODO")

    def update_obs_params_VB(self, SS, mergeCompA=None, **kwargs):
        if mergeCompA is None:
            for k in xrange(self.K):
                self.comp[k] = self.obsPrior.get_post_distr(SS, k)
        else:
            self.comp[mergeCompA] = self.obsPrior.get_post_distr(SS, mergeCompA)


    def update_obs_params_soVB( self, SS, rho, **kwargs):
        ''' Grab Dirichlet posterior for lambda and perform stochastic update
        '''
        for k in xrange(self.K):
            Dstar = self.obsPrior.get_post_distr(SS, k)
            self.comp[k].post_update_soVB(rho, Dstar)

    def set_global_params(self, hmodel=None, topics=None, 
                                Etopics=None, **kwargs):
        ''' Set global params to provided values

            Params
            --------
            topics : K x V matrix, each row has positive reals that sum to one
                     topics[k,v] = probability of word v under topic k
        '''
        if hmodel is not None:
            self.K = hmodel.obsModel.K
            self.comp = copy.deepcopy(hmodel.obsModel.comp)
            return
        if Etopics is not None:
            topics = Etopics
        assert topics is not None
        self.K = topics.shape[0]
        self.comp = list()

        for k in range(self.K):
            # Scale up Etopics to lamvec, a V-len vector of positive entries,
            #   such that (1) E[phi] is still Etopics, and
            #             (2) lamvec = obsPrior.lamvec + [some suff stats]
            #   where (2) means that lamvec is a feasible posterior value
            ii = np.argmin(topics[k,:])
            lamvec = self.obsPrior.lamvec[ii]/topics[k,ii] * topics[k,:]
            # Cut-off values that are way way too big
            lamvec = np.minimum(lamvec, 1e9)
            self.comp.append(DirichletDistr(lamvec))

  ######################################################### Evidence
  #########################################################
    def calc_evidence(self, Data, SS, LP):
        if self.inferType == 'EM':
            return 0 # handled by alloc model
        # Calculate p(w | z, lambda) + p(lambda) - q(lambda)
        elbo_pWords = self.E_log_pW(SS)
        elbo_pLambda = self.E_log_pLambda()
        elbo_qLambda = self.E_log_qLambda()
        return elbo_pWords + elbo_pLambda - elbo_qLambda
        
    def E_log_pW(self, SS):
        ''' Calculate "data" term of the ELBO,
                E_{q(Z), q(Phi)} [ log p(X) ]

            which can be computed quickly as
              for v in range(VocabSize):
                for k in range(K):
                    lpw += effectiveCount(word v in topic k) * Elogphi[k,v]
            NOTE: ampFactor has already been applied to SS.WordCounts!
        '''
        Elogphi = self.getElogphiMatrix()
        lpw = np.sum(SS.WordCounts * Elogphi)
        return lpw
 
    def E_log_pLambda(self):
        logNormC = -1 * self.obsPrior.get_log_norm_const()
        logDirPDF = np.dot(self.getElogphiMatrix(), self.obsPrior.lamvec - 1.)
        return np.sum(logDirPDF + logNormC)
        
    def E_log_qLambda(self):
        ''' Return negative entropy!'''    
        lp = np.zeros(self.K)
        for k in xrange(self.K):
            lp[k] = self.comp[k].get_entropy()
        return -1*lp.sum()



  ######################################################### I/O Utils
  #########################################################  for humans
    def get_info_string(self):
        return 'Multinomial distribution'
  
    def get_info_string_prior(self):
        if self.obsPrior is None:
            return 'None'
        else:
            lamvec = self.obsPrior.lamvec
            if np.allclose(lamvec, lamvec[0]):
                return 'Symmetric Dirichlet, lambda=%.2f' % (lamvec[0])
            else:
                return 'Dirichlet, lambda %s' % (np2flatstr(lamvec[:3]))


  ######################################################### I/O Utils
  #########################################################  for machines
    def get_prior_dict( self ):
        PDict = self.obsPrior.to_dict()
        return PDict
