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
    def __init__(self, inferType, dataAtomType='word', obsPrior=None):
        self.inferType = inferType
        self.obsPrior = obsPrior
        self.dataAtomType = dataAtomType
        self.comp = list()

    @classmethod
    def CreateWithPrior(cls, inferType, priorArgDict, Data):
        if inferType == 'EM':
            raise NotImplementedError('TODO')
        else:
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        return cls(inferType, obsPrior=obsPrior)

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

    def setupWithAllocModel(self, allocModel):
      '''
      '''
      if type(allocModel) != str:
        allocModel = str(type(allocModel))
      if allocModel.lower().count('hdp') or allocModel.lower().count('admix'):
        self.setDataAtomType('word')
      else:
        self.setDataAtomType('doc')

    def setDataAtomType(self, dataAtomType):
      self.dataAtomType = dataAtomType

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
        elif self.dataAtomType == 'doc':
            DocWordMat = Data.to_sparse_docword_matrix()
            LP['E_log_soft_ev'] = self.E_logsoftev_DocData(DocWordMat)
        else:
            LP['E_logsoftev_WordsData'], L = self.E_logsoftev_WordsData(Data)
            LP['topics'] = L
        return LP

    def E_logsoftev_DocData(self, DocWordMat):
      ''' Return log soft evidence probabilities for each document
      '''
      Elogphi = self.getElogphiMatrix().T # V x K matrix
      return DocWordMat * Elogphi # D x K matrix
          

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
        return E_logsoftev_words, Elogphi
  

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
        if 'hard_asgn' in LP:
          Nmat = LP['hard_asgn'] # N x K
          BMat = Data.to_sparse_matrix(doBinary=True) # V x N 
          TopicWordCounts = (BMat * Nmat).T # matrix-matrix product
        elif self.dataAtomType == 'doc':
          DocWordMat = Data.to_sparse_docword_matrix() # D x V
          TopicWordCounts = LP['resp'].T * DocWordMat # mat-mat product
        else:
          wv = LP['word_variational']  # N x K
          WMat = Data.to_sparse_matrix() # V x N
          TopicWordCounts = (WMat * wv).T # matrix-matrix product

        SS.setField('WordCounts', TopicWordCounts, dims=('K','D'))

        if self.dataAtomType == 'word':
          SS.setField('N', np.sum(TopicWordCounts,axis=1), dims=('K'))
        return SS

  ######################################################### Global Params
  #########################################################   M-step

    def update_obs_params_EM(self, SS, **kwargs):
        raise NotImplementedError("TODO")

    def update_obs_params_VB(self, SS, mergeCompA=None, comps=None, **kwargs):
        if mergeCompA is None:
            if comps is None:
              comps = xrange(self.K)
            for k in comps:
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
                                wordcountTotal=1000.0,
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
            if np.min(Etopics[-1]) == np.max(Etopics[-1]):
              # Skip the last topic, since it wasn't actively assigned
              topics = Etopics[:-1]
            else:
              topics = Etopics
        assert topics is not None
        self.K = topics.shape[0]
        self.comp = list()
        wc = wordcountTotal / float(self.K)
        for k in range(self.K):
          lamvec = topics[k,:] * wc + self.obsPrior.lamvec
          self.comp.append(DirichletDistr(lamvec))          


  ######################################################### Evidence
  #########################################################
    def calc_evidence(self, Data, SS, LP, todict=False):
        if self.inferType == 'EM':
            return 0 # handled by alloc model
        # Calculate p(w | z, lambda) + p(lambda) - q(lambda)
        elbo_pWords = self.E_log_pW(SS)
        elbo_pLambda = self.E_log_pLambda()
        elbo_qLambda = self.E_log_qLambda()
        if todict:
          return dict(data_Elogp=elbo_pWords,
                      phi_Elogp=elbo_pLambda,
                      phi_Elogq=elbo_qLambda,
                     )
        return elbo_pWords + elbo_pLambda - elbo_qLambda
        
    def E_log_pW(self, SS):
        ''' Calculate "data" term of the ELBO,
                E_{q(Z), q(Phi)} [ log p(X) ]
        '''
        Elogphi = self.getElogphiMatrix()  # K x V
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
