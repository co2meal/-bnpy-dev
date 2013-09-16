'''
StochasticOnlineVBLearnAlg.py

Implementation of stochastic online VB (soVB) for bnpy models
'''
import numpy as np
from bnpy.learn import LearnAlg

class StochasticOnlineVBLearnAlg(LearnAlg):

  def __init__(self, **kwargs):
    super(type(self),self).__init__(**kwargs)
    self.rhodelay = self.algParams['rhodelay']
    self.rhoexp = self.algParams['rhoexp']

  def fit(self, hmodel, DataIterator):
    self.set_start_time_now()
    LP = None
    rho = 1.0
    nObsTotal = DataIterator.nObsTotal
    lapFracPerBatch = DataIterator.nObsBatch / float(nObsTotal)
    iterid = -1
    lapFrac = 0
    while DataIterator.has_next_batch():
      # Grab new data and update counts
      Dchunk = DataIterator.get_next_batch()
      iterid += 1
      lapFrac = (iterid+1) * lapFracPerBatch

      # M step with learning rate
      if iterid > 0:
        rho = (iterid + self.rhodelay) ** (-1.0 * self.rhoexp)
        hmodel.update_global_params(SS, rho)

      # E step
      LP = hmodel.calc_local_params(Dchunk, LP)
      SS = hmodel.get_global_suff_stats(Dchunk, LP, Ntotal=nObsTotal)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Dchunk, SS, LP)      

      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)
    
    #Finally, save, print and exit
    self.save_state(hmodel,iterid, lapFrac, evBound, doFinal=True) 
    status = "all data processed."
    self.print_state(hmodel,iterid, lapFrac, evBound, doFinal=True, status=status)
    return LP