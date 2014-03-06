import numpy as np

from .HDPModel import HDPModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil

import scipy.sparse
import logging
Log = logging.getLogger('bnpy')

class HDPHardMult(HDPModel):

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, 
                              nCoordAscentItersLP=20, convThrLP=0.01,   
                              doOnlySomeDocsLP=True, **kwargs):
    ''' Calculate document-specific quantities (E-step) using hard assignments.

        Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
                 q(word_variational | word_token_variables)
            (2) Approx posterior on doc-topic probabilities
                 q(doc_variational | document_topic_variables)
        Returns
        -------
        LP : local params dict
    '''
    # First, run soft assignments for nCoordAscentIters
    LP = self.calc_local_params_fast(Data, LP, 
                                        nCoordAscentItersLP,
                                        convThrLP,
                                        doOnlySomeDocsLP,
                                     )
    # Next, find hard assignments
    LP['hard_mult'] = NumericHardUtil.findMode_Multinomial(
                                      Data.word_count,
                                      LP['word_variational']
                                      )    

    for d in xrange(Data.nDoc):
      start = Data.doc_range[d,0]
      stop = Data.doc_range[d,1]
      LP['DocTopicCount'][d,:] = np.sum(LP['hard_mult'][start:stop], axis=0)

    # Update doc_variational field of LP
    LP = self.get_doc_variational(Data, LP)
    LP = self.calc_ElogPi(LP)

    return LP


  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                              doPrecompMergeEntropy=False,
                                              mPairIDs=None):
    ''' Count expected number of times each topic is used across all docs    
    '''
    # Turn dim checking off, since some stats have dim K+1 instead of K
    SS = SuffStatBag(K=K, D=Data.vocab_size)
    SS.setField('nDoc', Data.nDoc, dims=None)
    sumLogPi = np.sum(LP['E_logPi'], axis=0)
    SS.setField('sumLogPiActive', sumLogPi[:K], dims='K')
    SS.setField('sumLogPiUnused', sumLogPi[-1], dims=None)

    if doPrecompEntropy:
      # ---------------- Z terms
      SS.setELBOTerm('ElogpZ', self.E_logpZ(Data, LP), dims='K')
      SS.setELBOTerm('ElogfactorialZ', self.E_logfactorialZ(Data, LP), dims='K')

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

      SS.setELBOTerm('ElogfactorialZ', 
                     self.E_logfactorialZ_merge(LP),
                     dims=('K', 'K'))
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
      E_logfactorialZ = np.sum(SS.getELBOTerm('ElogfactorialZ'))
    else:
      E_logqPi = self.E_logqPi(LP)
      E_logpZ = np.sum(self.E_logpZ(Data, LP))
      E_logfactorialZ = np.sum(self.E_logfactorialZ(Data, LP))

    if SS.hasAmpFactor():
      E_logqPi *= SS.ampF
      E_logpZ *= SS.ampF

    elbo = E_logpPi - E_logqPi 
    elbo += E_logpZ - E_logfactorialZ
    elbo -= E_logpV - E_logqV
    return elbo

