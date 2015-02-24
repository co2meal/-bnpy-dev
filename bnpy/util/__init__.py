"""
The :mod:`util` module gathers utility functions
"""

import RandUtil

from .PrettyPrintUtil import np2flatstr, flatstr2np
from .MatMultUtil import dotATA, dotATB, dotABT
from .RandUtil import discrete_single_draw, discrete_single_draw_vectorized
from .RandUtil import choice
from .SpecialFuncUtil import MVgammaln, MVdigamma, digamma, gammaln
from .SpecialFuncUtil import LOGTWO, LOGPI, LOGTWOPI, EPS
from .SpecialFuncUtil import logsumexp
from .VerificationUtil import isEvenlyDivisibleFloat, assert_allclose
from .ShapeUtil import as1D, as2D, as3D, toCArray

__all__ = ['RandUtil',
           'np2flatstr', 'flatstr2np',
           'dotATA', 'dotATB', 'dotABT',
           'discrete_single_draw',
           'MVgammaln', 'MVdigamma', 'logsumexp', 'digamma', 'gammaln',
           'isEvenlyDivisibleFloat', 'assert_allclose',
           'LOGTWO', 'LOGTWOPI', 'LOGPI', 'EPS',
           'as1D', 'as2D', 'as3D', 'toCArray']
