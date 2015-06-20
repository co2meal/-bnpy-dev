"""
The :mod:`util` module gathers utility functions
  for IO, special functions like "logsumexp", 
  and various random sampling functions
"""

import RandUtil

from .IOUtil import np2flatstr, flatstr2np, np2strList
from .LinAlgUtil import dotATA, dotATB, dotABT
from .RandUtil import discrete_single_draw, discrete_single_draw_vectorized
from .RandUtil import choice
from .SpecialFuncUtil import MVgammaln, MVdigamma, digamma, gammaln
from .SpecialFuncUtil import LOGTWO, LOGPI, LOGTWOPI, EPS
from .SpecialFuncUtil import logsumexp
from .VerificationUtil import closeAtMSigFigs, isEvenlyDivisibleFloat
from .ShapeUtil import as1D, as2D, as3D

__all__ = ['RandUtil', 
           'np2flatstr', 'flatstr2np', 'np2strList',
           'dotATA', 'dotATB', 'dotABT', 
           'discrete_single_draw', 
           'MVgammaln', 'MVdigamma', 'logsumexp', 'digamma', 'gammaln',
           'closeAtMSigFigs', 'isEvenlyDivisibleFloat',
           'LOGTWO', 'LOGTWOPI', 'LOGPI', 'EPS',
           'as1D', 'as2D', 'as3D']
