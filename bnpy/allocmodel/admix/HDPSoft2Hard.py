
import numpy as np

from .HDPModel import HDPModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil, NumericHardUtil
import LocalStepBagOfWords

import scipy.sparse
import logging
Log = logging.getLogger('bnpy')

class HDPSoft2Hard(HDPModel):

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, nHardItersLP=1, **kwargs):
    ''' Calculate document-specific quantities (E-step) using hard assignments.

        Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
                 q(word_variational | word_token_variables)
            (2) Approx posterior on doc-topic probabilities
                 q(doc_variational | document_topic_variables)

        Finishes with *hard* assignments!

        Returns
        -------
        LP : local params dict, with fields
            Pi : nDoc x K+1 matrix, 
                  row d has params for doc d's Dirichlet over K+1 topics
            word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
            DocTopicCount : nDoc x K matrix
    '''
    # First, run soft assignments for nCoordAscentIters
    LP = self._calc_local_params_fast(Data, LP, **kwargs)

    # Next, finish with hard assignments
    for rep in xrange(nHardItersLP):
      LP['word_variational'] = NumericHardUtil.toHardAssignmentMatrix(
                                                    LP['word_variational'])

      LP = LocalStepBagOfWords.update_DocTopicCount(Data, LP)
      LP = LocalStepBagOfWords.update_theta(LP, self.gamma*self.Ebeta[:-1],
                                                self.gamma*self.Ebeta[-1])
      LP = LocalStepBagOfWords.update_ElogPi(LP, self.gamma*self.Ebeta[-1])

    return LP

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                              doPrecompMergeEntropy=False,
                                              mPairIDs=None):
    ''' Count expected number of times each topic is used across all docs    
    '''
    wv = LP['word_variational']
    _, K = wv.shape
    # Turn dim checking off, since some stats have dim K+1 instead of K
    SS = SuffStatBag(K=K, D=Data.vocab_size)
    SS.setField('nDoc', Data.nDoc, dims=None)
    sumLogPi = np.sum(LP['E_logPi'], axis=0)
    SS.setField('sumLogPiActive', sumLogPi[:K], dims='K')
    SS.setField('sumLogPiUnused', sumLogPi[-1], dims=None)

    if 'DocTopicFrac' in LP:
      Nmajor = LP['DocTopicFrac']
      Nmajor[Nmajor < 0.05] = 0
      SS.setField('Nmajor', np.sum(Nmajor, axis=0), dims='K')
    if doPrecompEntropy:
      # ---------------- Z terms
      SS.setELBOTerm('ElogpZ', self.E_logpZ(Data, LP), dims='K')
      # ---------------- Pi terms
      # Note: no terms needed for ElogpPI
      # SS already has field sumLogPi, which is sufficient for this term
      ElogqPiC, ElogqPiA, ElogqPiU = self.E_logqPi_Memoized_from_LP(LP)
      SS.setELBOTerm('ElogqPiConst', ElogqPiC, dims=None)
      SS.setELBOTerm('ElogqPiActive', ElogqPiA, dims='K')
      SS.setELBOTerm('ElogqPiUnused', ElogqPiU, dims=None)

    if doPrecompMergeEntropy:
      ElogpZMat, sLgPiMat, ElogqPiMat = self.memo_elbo_terms_for_merge(LP)
      SS.setMergeTerm('ElogpZ', ElogpZMat, dims=('K','K'))
      SS.setMergeTerm('ElogqPiActive', ElogqPiMat, dims=('K','K'))
      SS.setMergeTerm('sumLogPiActive', sLgPiMat, dims=('K','K'))
    return SS

        
  ######################################################### Evidence
  #########################################################  
  def calc_evidence( self, Data, SS, LP ):
    ''' Calculate ELBO terms related to allocation model
    '''   
    E_logpV = self.E_logpV()
    E_logqV = self.E_logqV()
    
    E_logpPi = self.E_logpPi(SS)
    if SS.hasELBOTerms():
      E_logqPi = SS.getELBOTerm('ElogqPiConst') \
                  + SS.getELBOTerm('ElogqPiUnused') \
                  + np.sum(SS.getELBOTerm('ElogqPiActive'))
      E_logpZ = np.sum(SS.getELBOTerm('ElogpZ'))
    else:
      E_logqPi = self.E_logqPi(LP)
      E_logpZ = np.sum(self.E_logpZ(Data, LP))

    if SS.hasAmpFactor():
      E_logqPi *= SS.ampF
      E_logpZ *= SS.ampF


    elbo = E_logpPi - E_logqPi
    elbo += E_logpZ
    elbo += E_logpV - E_logqV
    return elbo


