'''
  MixModel.py
     Bayesian parametric mixture model with a finite number of components K

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   K        : # of components
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

'''
import numpy as np

from bnpy.allocmodel import AllocModel

from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS

class DPMixModel(AllocModel):
  pass