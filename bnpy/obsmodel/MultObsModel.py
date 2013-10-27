'''
'''
import numpy as np

from ..distr import DirichletDistr
from ..distr import MultinomialDistr
from ..util import np2flatstr, EPS

from ObsCompSet import ObsCompSet

class MultObsModel( ObsCompSet ):
    
    def __init__(self, inferType, obsPrior=None):
        self.inferType = inferType
        self.obsPrior = obsPrior
        self.comp = list()

    @classmethod
    def InitFromCompDicts(cls, oDict, obsPrior, compDictList):
        ''' Create MultObsCompSet, all K component Distr objects, 
            and the prior Distr object in one call
        '''
        self = cls(oDict['inferType'], obsPrior=obsPrior)
        self.K = len(compDictList)
        self.comp = [None for k in range(self.K)]
        for k in xrange(self.K):
            if oDict['inferType'] == 'EM':
                raise NotImplementedError("TODO")
            else:
                self.comp[k] = DirichletDistr(**compDictList[k]) 
        return self

    @classmethod
    def InitFromData(cls, inferType, priorArgDict, Data):
        # Something else probably needs to be done for EM here
        if inferType == 'EM':
            # Defines the Dirichlet (topic x word) that is initialized to ensure it fits data dimensions
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        else:
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        return cls(inferType, obsPrior)

    def get_human_global_param_string(self, fmtStr='%3.2f'):
        if self.inferType == 'EM':
            return '\n'.join( [np2flatstr(self.obsPrior[k].phi, fmtStr) for k in xrange(self.K)] )
        else:
            return '\n'.join( [np2flatstr(self.obsPrior[k].lamvec/self.obsPrior[k].lamsum, fmtStr) for k in xrange(self.K)] )

    def set_global_params(self, true_tw=None, mass=100, **kwargs):
        ''' Set global params to provided values
        '''
        self.K = true_tw.shape[0]
        self.comp = list()
        for k in range(self.K):
          self.comp.append(DirichletDistr(mass * true_tw[k,:]))

    def get_global_suff_stats(self, Data, SS, LP, **kwargs):
        ''' Calculate and return sufficient statistics.

            Returns
            -------
            SS : bnpy SuffStatDict object, with updated fields
                WordCounts : K x VocabSize matrix
                  WordCounts[k,v] = # times vocab word v seen with topic k
        '''
        # Grab topic x word sufficient statistics
        wv = LP['word_variational']
        nDistinctWords, K = wv.shape

        # IMPROVED: sparse matrix product!
        WMat = Data.to_sparse_matrix()
        TopicWordCounts = (WMat * wv).T

        # OLD WAY: for loop
        #TopicWordCounts = np.zeros((K, Data.vocab_size)) 
        #effCount = wv * Data.word_count[:, np.newaxis]
        #for ii in xrange(nDistinctWords):
        #    TopicWordCounts[:, Data.word_id[ii]] += effCount[ii]
        SS.WordCounts = TopicWordCounts
        SS.N = np.sum(SS.WordCounts, axis=1)
        return SS

    def update_obs_params_EM(self, SS, **kwargs):
        raise NotImplementedError("TODO")

    def update_obs_params_VB(self, SS, Krange, **kwargs):
        for k in Krange:
            self.comp[k] = self.obsPrior.get_post_distr(SS.getComp(k))

    def update_obs_params_soVB( self, SS, rho, Krange, **kwargs):
        # grab Dirichlet posterior for lambda and perform stochastic update
        for k in Krange:
            Dstar = self.obsPrior.get_post_distr(SS.getComp(k))
            self.comp[k].post_update_soVB(rho, Dstar)
      
    def calc_local_params( self, Data, LP):
        ''' Calculate local parameters (E-step)
            For LDA, these are expectations for assigning each observed word
            to all K possible topics.

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
  
    def getElogphiMatrix(self):
      Elogphi = np.empty((self.K, self.comp[0].D))
      for k in xrange(self.K):
        Elogphi[k,:] = self.comp[k].Elogphi
      return Elogphi

  ######################################################### Evidence Bound Fcns  
    def calc_evidence(self, Data, SS, LP):
        if self.inferType == 'EM':
            return 0 # handled by alloc model
        # Calculate p(w | z, lambda) + p(lambda) - q(lambda)
        elbo_pWords = self.E_log_pW(SS)
        elbo_pLambda = self.E_log_pLambda()
        elbo_qLambda = self.E_log_qLambda()
        lb_obs = elbo_pWords + elbo_pLambda - elbo_qLambda
        
        return lb_obs
  
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

    def E_log_pW_matrixproduct(self, Data, LP, SS):
        ''' DEPRECATED, used only to check calculations
            Calculates E_{q(Z), q(Phi)} [ log p(X) ]
        '''        
        # Obtain matrix where column k = E[ log phi[k] ], for easier indexing
        Elogphi = self.getElogphiMatrix().T

        lpw = np.sum(Elogphi[wid,:] * LP['word_variational'], axis=1)
        lpw = np.inner(lpw, Data.word_count)
        if SS.hasAmpFactor():
          return SS.ampF * lpw
        else:
          return lpw

    def E_log_pW_forloop(self, Data, LP, SS):
        ''' DEPRECATED! used only to double-check calculations.
            Calculates E_{q(Z), q(Phi)} [ log p(X) ]
            Loops over every word in corpus, so definitely too slow
        '''
        Elambda_kw = np.zeros((self.K, Data.vocab_size))
        for k in xrange(self.K):
            Elambda_kw[k,:] = self.comp[k].Elogphi
        lpw = 0
        for ii in xrange( Data.nObs ):
            lpw += Data.word_count[ii] * np.dot(LP["word_variational"][ii,:] ,Elambda_kw[:, Data.word_id[ii]]) 
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

    def get_prior_dict( self ):
        PDict = self.obsPrior.to_dict()
        return PDict

############### GET METHODS ##################
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
                return 'Dirichlet, lambda %s' % (np2flatstr(lamvec))
