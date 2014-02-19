"""
The :mod:`mix` module gathers point-estimate and variational approximations
   for Bayesian mixture modeling, including
      finite parametric mixture models
      nonparametric Dirichlet Process and Pitman-Yor mixture models
"""

from MixModel import MixModel
from DPMixModel import DPMixModel
__all__ = ['MixModel', 'DPMixModel']