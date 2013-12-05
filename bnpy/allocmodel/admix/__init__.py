"""
The :mod:`mix` module gathers point-estimate and variational approximations
   for Bayesian mixture modeling, including
      finite parametric mixture models
      nonparametric Dirichlet Process and Pitman-Yor mixture models
"""

from AdmixModel import AdmixModel
from HDPModel import HDPModel
#import HDPVariationalOptimizer

__all__ = ['AdmixModel', 'HDPModel']
